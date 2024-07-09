import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_atmosphere(image, scatterlight):
    for _ in range(1):
        scatter_est = np.sum(scatterlight, axis=2)
        n_pixels = scatter_est.size
        n_search_pixels = int(n_pixels * 0.001)
        image_vec = image.reshape(n_pixels, 3)
        indices = np.argsort(scatter_est.flatten())[::-1]
        atmosphere = np.mean(image_vec[indices[:n_search_pixels], :], axis=0)

        atmosphere = np.repeat(
            atmosphere[np.newaxis, np.newaxis, :], scatter_est.shape[0], axis=0
        )
        atmosphere = np.repeat(atmosphere, scatter_est.shape[1], axis=1)

        sek = scatter_est.flatten()[indices[n_search_pixels]]

        mask = scatter_est <= sek
        scatterlight = scatterlight * np.repeat(mask[:, :, np.newaxis], 3, axis=2) + (
            2 / 3 * sek - scatterlight
        ) * np.repeat((~mask)[:, :, np.newaxis], 3, axis=2)

    return atmosphere, scatterlight


def scattering_mask_sample(I):
    mSize = min(I.shape[0], I.shape[1])
    if mSize < 800:
        r = 0.02
    elif 800 <= mSize < 1500:
        r = 0.01
    else:
        r = 0.005

    I0 = cv2.resize(I, (0, 0), fx=r, fy=r)
    return cv2.resize(I0, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_CUBIC)


def gamma0(img):
    i = np.arange(256)
    f = ((i + 0.5) / 256) ** (5 / 6)
    LUT = np.uint8(f * 256 - 0.5)

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img_ycrcb[:, :, 0] = LUT[img_ycrcb[:, :, 0]]
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)


def rank1_enhancement(img, omega=0.8):
    imgvec = img.reshape(-1, 3)
    x_RGB = np.mean(imgvec, axis=0)
    x_mean = np.repeat(x_RGB[np.newaxis, np.newaxis, :], img.shape[0], axis=0)
    x_mean = np.repeat(x_mean, img.shape[1], axis=1)

    scat_basis = x_mean / np.maximum(
        np.sqrt(np.sum(x_mean**2, axis=2, keepdims=True)), 0.001
    )
    fog_basis = img / np.maximum(np.sqrt(np.sum(img**2, axis=2, keepdims=True)), 0.001)
    cs_sim = np.sum(scat_basis * fog_basis, axis=2, keepdims=True)

    scattering_light = (
        cs_sim
        * (
            np.sum(img, axis=2, keepdims=True)
            / np.maximum(np.sum(x_mean, axis=2, keepdims=True), 0.001)
        )
        * x_mean
    )

    atmosphere, scattering_light = get_atmosphere(img, scattering_light)

    T = 1 - omega * scattering_light
    T_ini = scattering_mask_sample(T)

    ShowR_d = (img - atmosphere) / np.maximum(T_ini, 0.001) + atmosphere

    mi = np.percentile(ShowR_d, 1, axis=(0, 1))
    ma = np.percentile(ShowR_d, 99, axis=(0, 1))
    Jr_ini = (ShowR_d - mi) / (ma - mi)

    cv2.normalize(Jr_ini, Jr_ini, 0, 255, cv2.NORM_MINMAX)
    Jr_ini = np.uint8(Jr_ini)

    Jr_ini = gamma0(Jr_ini)

    tildeT = 1 - T
    s_tildeT = 1 - T_ini

    return Jr_ini, s_tildeT, tildeT


def main():
    import os

    imgs = ["hazy1.png"]
    for img in imgs:
        img_path = f"images/{img}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        J, s_tildeT, tildeT = rank1_enhancement(img)

        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(img)
        plt.title("Original Image")
        plt.subplot(122)
        plt.imshow(J)
        plt.title("Enhanced Image")
        plt.tight_layout()
        plt.show()

        if not os.path.exists("results"):
            os.makedirs("results")

        cv2.imwrite(f"results/{img}-Rank1.png", cv2.cvtColor(J, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"results/{img}-sT-Rank1.png", s_tildeT)
        cv2.imwrite(f"results/{img}-T-Rank1.png", tildeT)


if __name__ == "__main__":
    main()
