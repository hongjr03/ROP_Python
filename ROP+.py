import numpy as np
import cv2

# from icecream import ic
from tqdm import tqdm


def defaultParamSetting(ImageType):
    if ImageType == 1:
        param = {"omega": 0.99, "mi": 1, "ma": 95, "gamma": 5e1}
    elif ImageType == 2:
        param = {"omega": 0.8, "mi": 1, "ma": 95, "gamma": 3e1}
    elif ImageType == 3:
        param = {"omega": 0.75, "mi": 1, "ma": 95, "gamma": 5e1}
    elif ImageType == 4:
        param = {"omega": 0.7, "mi": 2, "ma": 99.6, "gamma": 1e2}
    param["iternum"] = 10

    return param


# 获得大气光和散射光
def get_atmosphere(image, tilde_T):
    scatter_est = np.sum(tilde_T, axis=2)
    n_pixels = scatter_est.size
    # ic(n_pixels)
    n_search_pixels = int(n_pixels * 0.01)
    image_vec = image.reshape(n_pixels, 3)
    indices = np.argsort(scatter_est.flatten())[::-1]
    atmosphere = np.mean(image_vec[indices[:n_search_pixels], :], axis=0)

    atmosphere = np.tile(
        atmosphere[np.newaxis, np.newaxis, :],
        (scatter_est.shape[0], scatter_est.shape[1], 1),
    )

    sek = scatter_est.flatten()[indices[n_search_pixels]]
    # ic(sek)
    sek_vec = np.tile(
        sek * tilde_T[0, 0] / np.maximum(scatter_est[0, 0], 0.001),
        (tilde_T.shape[0], tilde_T.shape[1], 1),
    )
    # ic(sek_vec.shape)

    tilde_T = tilde_T * np.repeat(scatter_est <= sek, 3).reshape(tilde_T.shape) + (
        2 * sek_vec - tilde_T
    ) * np.repeat(scatter_est > sek, 3).reshape(tilde_T.shape)

    return atmosphere, tilde_T


def TransRefine(atmosphere, I, t0, param):
    # C0 is the intensity of the transmission map T
    # SnuT is the unified spectrum of the transimssion map T
    # SnuT = T(1,1,:)/sum(T(1,1,:),3)

    def getCoeff():
        from psf2otf import psf2otf

        # from pypher.pypher import psf2otf

        N, M, C = t0.shape
        sizeF = np.array([N, M])
        # # ic(sizeF)
        Coeff = {
            "eigsD1": np.repeat(psf2otf(np.array([[1, -1]]), sizeF), 3),
            "eigsD2": np.repeat(psf2otf(np.array([[1], [-1]]), sizeF), 3),
            # "eigsDtD": np.abs(Coeff["eigsD1"]) ** 2 + np.abs(Coeff["eigsD2"]) ** 2,
            "eigsD21": psf2otf(np.array([[1, -1]]), [sizeF[0], sizeF[1]]),
            "eigsD22": psf2otf(np.array([[1], [-1]]), [sizeF[0], sizeF[1]]),
            # "eigsDtD2": np.abs(Coeff["eigsD21"]) ** 2 + np.abs(Coeff["eigsD22"]) ** 2,
        }
        Coeff["eigsDtD"] = (
            np.abs(Coeff["eigsD1"]) ** 2 + np.abs(Coeff["eigsD2"]) ** 2
        ).reshape([N, M, C])
        Coeff["eigsDtD2"] = (
            np.abs(Coeff["eigsD21"]) ** 2 + np.abs(Coeff["eigsD22"]) ** 2
        ).reshape([N, M])
        return Coeff

    def ForwardD(U):
        # % Forward finite difference operator
        end_col_diff = np.expand_dims((U[:, 0] - U[:, -1]), axis=1)
        end_row_diff = np.expand_dims((U[0, :] - U[-1, :]), axis=0)
        Dux = np.concatenate(
            (np.diff(U, 1, 1), end_col_diff), axis=1
        )  # discrete gradient operators
        Duy = np.concatenate((np.diff(U, 1, 0), end_row_diff), axis=0)
        return (Dux, Duy)

    def Dive(X, Y):
        # % Transpose of the forward finite difference operator
        fwd_diff_rowX = np.expand_dims(X[:, -1] - X[:, 1], axis=1)
        DtXY = np.concatenate((fwd_diff_rowX, -np.diff(X, 1, 1)), axis=1)
        fwd_diff_rowY = np.expand_dims(Y[-1, :] - Y[1, :], axis=0)
        DtXY = DtXY + np.concatenate((fwd_diff_rowY, -np.diff(Y, 1, 0)), axis=0)
        return DtXY

    Coeff = getCoeff()
    D, Dt = ForwardD, Dive
    MaxIter = param["iternum"]
    gamma = param["gamma"]

    lambda1 = 5
    lambda2 = 5e-1
    lambda3 = 5e-1
    beta = 10

    M, N, Channel = t0.shape
    Lxi1 = np.zeros((M, N, Channel))
    Lxi2 = np.zeros((M, N, Channel))
    Leta1 = np.zeros((M, N))
    Leta2 = np.zeros((M, N))

    SolRE = 5e-4
    tau = 1.618

    D1I, D2I = D(I)
    C0 = np.mean(t0, axis=2)

    SnuT = t0[0, 0, :] / np.sum(t0[0, 0, :].reshape(1, 1, 3), axis=2, keepdims=True)

    D1tG, D2tG = D(C0)
    # D1t = SnuT .* repmat(D1tG, [1 1 3]);
    # D2t = SnuT .* repmat(D2tG, [1 1 3]);
    D1t = SnuT * np.repeat(D1tG, 3).reshape(M, N, Channel)
    D2t = SnuT * np.repeat(D2tG, 3).reshape(M, N, Channel)

    C = C0

    cont = True
    k = 0

    while tqdm(cont, desc="Iterating", leave=False):
        k += 1

        Xterm1 = D1t - D1I + Lxi1 / beta
        Xterm2 = D2t - D2I + Lxi2 / beta
        Xterm = np.sqrt(Xterm1**2 + Xterm2**2)
        W = np.exp(-gamma * (np.abs(D1I) + np.abs(D2I)))
        Xterm = np.maximum(Xterm - (W * lambda1) / beta, 0) / (
            Xterm + np.finfo(float).eps
        )
        Xmg1 = Xterm * Xterm1
        Xmg2 = Xterm * Xterm2

        Zterm1 = D1tG + Leta1 / beta
        Zterm2 = D2tG + Leta2 / beta
        Zterm = np.sqrt(Zterm1**2 + Zterm2**2)
        Zterm = np.maximum(Zterm - lambda2 / beta, 0) / (Zterm + np.finfo(float).eps)
        Zmg1 = Zterm * Zterm1
        Zmg2 = Zterm * Zterm2

        zeta1X = Xmg1 + D1I - Lxi1 / beta
        zeta1Y = Xmg2 + D2I - Lxi2 / beta
        zeta2X = Zmg1 - Leta1 / beta
        zeta2Y = Zmg2 - Leta2 / beta

        from numpy.fft import fft2, ifft2

        ttem = (
            fft2(lambda3 * C0)
            + beta * np.sum(fft2(Dt(SnuT * zeta1X, SnuT * zeta1Y)), 2)
            + beta * fft2(Dt(zeta2X, zeta2Y))
        )
        ttemp = lambda3 + beta * (
            np.sum(SnuT**2 * Coeff["eigsDtD"], 2) + Coeff["eigsDtD2"]
        )
        Cnew = np.real(ifft2(ttem / (ttemp + np.finfo(float).eps)))
        Cnew[Cnew <= 0] = 0
        Cnew[Cnew >= 1] = 1

        D1tG, D2tG = D(Cnew)
        D1t = SnuT * np.repeat(D1tG, 3).reshape(M, N, Channel)
        D2t = SnuT * np.repeat(D2tG, 3).reshape(M, N, Channel)

        Lxi1 = Lxi1 - tau * beta * (Xmg1 - (D1t - D1I))
        Lxi2 = Lxi2 - tau * beta * (Xmg2 - (D2t - D2I))
        Leta1 = Leta1 - tau * beta * (Zmg1 - D1tG)
        Leta2 = Leta2 - tau * beta * (Zmg2 - D2tG)

        re = np.linalg.norm(Cnew - C, "fro") / np.linalg.norm(C, "fro")
        # ic(k, re)
        C = Cnew
        cont = (k < MaxIter) and (re > SolRE)

    # ic(SnuT.shape, C.shape)
    T = SnuT * np.repeat(C, 3).reshape(M, N, Channel)
    Jr = (I - atmosphere) / np.maximum(T, 0.01) + atmosphere

    return Jr, T


def gamma0(img):

    i = np.arange(256)
    f = ((i + 0.5) / 256) ** (5 / 6)
    LUT = np.uint8(f * 256 - 0.5)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img[:, :, 0] = LUT[img[:, :, 0]]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return img


def rank1plus_enhancement(img, ImageType):

    param = defaultParamSetting(ImageType)

    imgvec = img.reshape(img.shape[0], img.shape[1], 3)
    selectedpixel = np.ones_like(imgvec)
    previous_basis = np.ones(3).reshape(1, 3)

    for _ in range(20):
        # 计算 imgvec 三个通道的均值，得到 (1, 1, 1) 的数组
        x_RGB = np.mean(imgvec, axis=(0, 1))
        # 扩展到与 imgvec 相同的形状
        x_mean = np.tile(x_RGB, (img.shape[0], img.shape[1], 1))
        # normailize
        spec_basis = x_mean / np.maximum(
            np.sqrt(np.sum(x_mean**2, axis=2, keepdims=True)), 0.001
        )
        image_norm = img / np.maximum(
            np.sqrt(np.sum(img**2, axis=2, keepdims=True)), 0.001
        )
        # 计算相似度
        # proj_sim = repmat((((sum(spec_basis .* imag_nmlzed, 3)))), [1 1 3]);
        proj_sim = np.sum(spec_basis * image_norm, axis=2, keepdims=True)
        proj_sim = np.repeat(proj_sim, 3, axis=2)
        # 计算散射光
        if np.sum(np.abs(previous_basis - spec_basis)) != 0:
            # 更新 previous_basis
            previous_basis = spec_basis
            boots = proj_sim.reshape(img.shape[0] * img.shape[1], 3)
            selectedpixel = np.where(
                boots > 0.99, 1, 0
            )  # 如果 proj_sim 大于 0.99，就保留，否则置 0
        else:
            break

    # S_nu 是 unified_spectrum of \tilde{t}，即散射光的强度
    S_nu = x_mean / np.maximum(np.sum(x_mean, axis=2, keepdims=True), 0.001)
    tilde_T = proj_sim * np.sum(img, axis=2, keepdims=True) * S_nu

    initial_img = img

    atmosphere, tilde_T = get_atmosphere(initial_img, tilde_T)

    # T = 1 - omega * tilde_T
    T_ini = 1 - param["omega"] * tilde_T

    # ic(initial_img.shape, T_ini.shape)
    Jr, T_tv = TransRefine(atmosphere, initial_img, T_ini, param)

    mi = np.percentile(Jr, param["mi"], axis=(0, 1))
    ma = np.percentile(Jr, param["ma"], axis=(0, 1))
    Jr = (Jr - mi) / (ma - mi)
    cv2.normalize(Jr, Jr, 0, 255, cv2.NORM_MINMAX)
    Jr = np.uint8(Jr)
    Jr = gamma0(Jr)
    return Jr, T_tv, T_ini


def main():
    import os

    img_path = "images"
    imgs = os.listdir(img_path)
    for img in tqdm(imgs):
        img_path = f"images/{img}"
        image = cv2.imread(img_path)
        image = image / 255.0

        Jr, T_tv, T_ini = rank1plus_enhancement(image, 3)

        # Jr = cv2.cvtColor(Jr, cv2.COLOR_BGR2RGB)
        # T_tv = cv2.cvtColor(T_tv, cv2.COLOR_BGR2RGB)
        # T_ini = cv2.cvtColor(T_ini, cv2.COLOR_BGR2RGB)

        if not os.path.exists("results"):
            os.makedirs("results")
        img = img.split(".")[0]
        cv2.imwrite(f"results/{img}-rank1plus.png", Jr)
        cv2.imwrite(f"results/{img}-sT-rank1plus.png", 255 - T_tv)
        cv2.imwrite(f"results/{img}-T-rank1plus.png", 255 - T_ini)

        # print(f"Results saved to results/{img}-Rank1.png")


if __name__ == "__main__":
    main()
