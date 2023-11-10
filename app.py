import os
from flask import Flask, request, render_template, redirect, url_for
from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)


neon = [
    3718.2065,  # ArII
    # 3737.1313,  # FeI
    # 3820.4251,  # FeI
    # 3906.4794,  # FeI
    3925.7188,  # ArII
    3946.0971,  # ArII
    # 3969.2570,  # FeI
    4131.7233,  # ArII
    4158.5910,  # ArI
    # 4198.3036,  # FeI
    4201.9713,  # ArII(?)
    # 4302.1856,  # FeI
    # 4337.0454,  # FeI
    # 4351.5437,  # FeI(?)
    4379.6665,  # ArII
    # 4385.3768,  # FeII(?)
    # 4427.3095,  # FeI
    # 4482.1691,  # FeI
    4510.7332,  # ArI
    4545.0517,  # ArII
    4579.3493,  # ArII
    4589.8976,  # ArII
    4609.5673,  # ArII
    4657.9009,  # ArII
    4726.8681,  # ArII
    4764.8644,  # ArII
    4806.0201,  # ArII
    4847.8095,  # ArII
    4879.8634,  # ArII
    # 4957.5961,  # FeI blend with FeI 4957.2979
    4965.0794,  # ArII
    5017.1626,  # ArII
    5062.0370,  # ArII
    # 5110.4129,  # FeI
    5141.7826,  # ArII
    5145.3082,  # ArII
    # 5167.4879,  # FeI
    # 5187.7460,  # ArI blended with FeI 5187.9138
    5187.5782,
    # 5227.1889,  # FeI
    # 5269.5366,  # FeI
    # 5328.2846,  # blend FeI 5328.0381 with FeI 5328.5311
    5341.0932,  # NeI
    # 5371.4891,  # FeI
    5400.5618,  # NeI
    5451.6520,  # ArI
    5495.8730,  # ArI
    5506.1120,  # ArI
    5559.0956,  # NeI
    # 5572.8421,  # FeI
    5606.7330,  # ArI
    # 5615.6436,  # FeI
    5650.7040,  # ArI
    5656.6578,  # NeI
    5662.5476,  # NeI
    5689.8156,  # NeI
    5719.2256,  # NeI blended with NeI 5718.8798
    5739.5190,  # ArI
    5748.2979,  # NeI
    5764.4189,  # NeI
    5804.4496,  # NeI
    5820.1548,  # NeI
    5852.4879,  # NeI
    5881.8952,  # NeI
    5888.5840,  # ArI
    5902.4622,  # NeI
    5912.0850,  # ArI blended with NeI 5913.6330
    5944.8342,  # NeI
    5975.5340,  # NeI blended with NeI 5974.6282
    6029.9969,  # NeI
    6032.1270,  # ArI
    6043.2230,  # ArI
    6074.3377,  # NeI
    6096.1631,  # NeI
    6114.9232,  # ArII
    6143.0626,  # NeI
    6163.5939,  # NeI
    #6172.2775,  # ArI blended with doublet FeIII (6169.7260 and 6169.7500)
    6177.3575,
    6182.1463,  # NeI
    # 6213.8757,  # NeI blended with ArI 6212.5030 and FeI 6213.4291
    6214.099,
    6217.2812,  # NeI
    6266.4950,  # NeI
    6304.7889,  # NeI
    6334.4278,  # NeI
    6382.9917,  # NeI
    6402.2472,  # NeI
    6416.3070,  # ArI
    6506.5281,  # NeI
    6532.8822,  # NeI
    6598.9529,  # NeI
    6678.2762,  # NeI
    6717.0430,  # NeI
    # 6752.7704,  # blend FeI 6752.7067 with ArI 6752.8340
    6752.8340,
    6871.2891,  # ArI
    6929.4673,  # NeI
    6937.6640,  # ArI
    6965.4300,  # ArI
    7024.0504,  # NeI
    # 7024.2458,  # blend ArI 7030.2510 with FeII 7030.2405
    7030.2510,
    7032.4131,  # NeI
    7059.1074,  # NeI
    7067.2181,  # ArI
    7147.0410,  # ArI
    7206.9804,  # ArI
    7245.1666,  # NeI
    7272.9350,  # ArI
    7353.3000,  # ArI
    7372.1170,  # ArI
    7383.9800,  # ArI
    7503.8680,  # ArI
    7514.6510,  # ArI
    7635.1050,  # ArI
    7723.9835,  # blend ArI 7723.7600 with ArI 7724.2070
    7948.1764,  # ArI blend ? (no blend in NIST DB)
    8264.5225,  # ArI
    8408.2096,  # ArI
    8424.6475,  # ArI
    8667.9442,  # ArI
    8521.4422,  # ArI
    9122.9674,  # ArI
    9224.4992,  # ArI
    9354.2198,  # ArI
    9657.7863,  # ArI
    9784.5028,  # ArI
    10470.0535  # ArI
]
def find_local_maximums(arr):
    local_maximums = []

    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i]>3*np.std(arr):
            local_maximums.append(i)

    return local_maximums
def load_fits_data(file):
    try:
        hdul = fits.open(file)
        data_extension = next((i for i, hdu in enumerate(hdul) if hdu.data is not None), None)
        if data_extension is not None:
            data = hdul[data_extension].data
            return data
        else:
            return None
    except Exception as e:
        print("Error reading FITS file:", str(e))
        return None
def plot_spectrum(wavelength_values, spectrum):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(wavelength_values, spectrum, label='Spectrum', color='blue')
    ax.set_xlabel('Wavelength Units')
    ax.set_ylabel('Intensity Units')
    ax.set_title('Spectrum Plot')
    ax.legend()
    ax.grid(True)
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png')
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer

def master(a):
    stacked = np.stack(a)
    mean = np.mean(stacked, axis=0)
    # mean[mean > (np.mean(mean) + 10 * np.std(mean))] = np.mean(mean)
    # mean[mean < (np.mean(mean) - 10 * np.std(mean))] = np.mean(mean)
    median = np.median(stacked, axis=0)
    # median[median > (np.median(median) + 10 * np.std(median))] = np.median(median)
    # median[median < (np.median(median) - 10 * np.std(median))] = np.median(median)
    if np.std(mean) < np.std(median):
        return mean
    else:
        return median

@app.route('/', methods=['GET', 'POST'])
def process_fits_files():
    if request.method == 'POST':
        object_files = request.files.getlist('object_file')
        dark_files = request.files.getlist('dark_file')
        flat_files = request.files.getlist('flat_file')
        dflat_files = request.files.getlist('dflat_file')
        lamp_files = request.files.getlist('comp_lamp')

        object_data = []
        dark_data = []
        flat_data = []
        dflat_data = []
        lamp_data = []
        for object_file in object_files:
            data = load_fits_data(object_file)
            if data is not None:
                print('data= ',data.ndim)
                object_data.append(data)

        for dark_file in dark_files:
            data = load_fits_data(dark_file)
            if data is not None:
                dark_data.append(data)

        for flat_file in flat_files:
            data = load_fits_data(flat_file)
            if data is not None:
                flat_data.append(data)

        for dflat_file in dflat_files:
            data = load_fits_data(dflat_file)
            if data is not None:
                dflat_data.append(data)

        for lamp_file in lamp_files:
            data = load_fits_data(lamp_file)
            if data is not None:
                lamp_data.append(data)

        if len(object_data) > 0:
            object_data = master(np.stack(object_data))
            reduced = object_data
        if len(dark_data) > 0:
            dark_data = np.stack(dark_data)
            average_dark = master(dark_data)
            reduced = object_data - average_dark
        if len(flat_data) > 0:
            flat_data = np.stack(flat_data)
            average_flat = master(flat_data)
            calib = average_flat
            normalize = calib / np.mean(calib)
            reduced = (object_data - average_dark) / normalize
        if len(dflat_data) > 0:
            dflat_data = np.stack(dflat_data)
            average_dflat = master(dflat_data)
            calib = average_dflat - average_flat
            normalize = calib / np.mean(calib)
            reduced = (object_data - average_dark) / normalize
        if len(lamp_data) > 0:
            lamp_temp = master(np.stack(lamp_data))[:,::-1]
            wavelength_values = np.arange(lamp_temp.shape[1]) * 1
            spectrumc = np.sum(lamp_temp, axis=0)
            lamp=find_local_maximums(spectrumc)
            korder=[]
            for k in range(len(neon)-len(lamp)):
              sub=neon[k:k+len(lamp)]
              coefficients, residuals, _, _, _ = np.polyfit(lamp, sub, 2, full=True)
              korder.append(residuals)
            k = korder.index(min(korder))
            subfix=neon[k:k+len(lamp)]
            coefficients, residuals, _, _, _ = np.polyfit(lamp, subfix, 2, full=True)
            def f(x):
                return coefficients[0] * x**2 + coefficients[1] * x + coefficients[2]
            real_wavelength=f(wavelength_values)
            # sumar=[]
            sumred=[]
            new_x=np.linspace(real_wavelength[0],real_wavelength[len(real_wavelength)-1],1000)
            reduced=reduced[:,::-1]
            for i in range(len(lamp_temp[:][:])):
                try:
                    x=find_local_maximums(lamp_temp[:][i])
                    coefficients = np.polyfit(x, subfix, 2)
                    def f(a):
                        return coefficients[0] * a**2 + coefficients[1] * a + coefficients[2]
                except:
                    print(len(x), len(subfix))
                wavelength_values = (np.arange(lamp_temp.shape[1]) * 1)
                rw=f(wavelength_values)
                # new_y = np.interp(new_x, rw, lamp_temp[:][i])
                new_reduced = np.interp(new_x, rw, reduced[:][i])
                # sumar.append(new_y)
                sumred.append(new_reduced)
            spectrumrd = np.sum(sumred, axis=0)
            img_buffer = plot_spectrum(new_x, spectrumrd)
            img_data_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        if len(lamp_data) == 0:
            spectrum = np.sum(reduced, axis=0)[::-1]
            wavelength_values = np.arange(reduced.shape[1]) * 1
            img_buffer = plot_spectrum(wavelength_values, spectrum)
            img_data_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        print(subfix)
        print(lamp)
        return render_template('result.html', img_data=img_data_base64)

    return render_template('upload.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

