# solarcell
Estimate IV- and PV-curves given photovoltaic cell datasheet parameters. Capable of generating combined curves for solar arrays with temperature and light intensity gradients. Assumes ideal bypass and blocking diodes for all cells.

`pip install solarcell`

## Usage

Please see the examples contained within this repository.

```python
import numpy as np
from matplotlib import pyplot as plt
from solarcell import solarcell

# Azur Space 3G30A triple-junction solar cells in a 24s12p configuration.
# Isc/Imp are specified in (A, A/C), Voc/Vmp are specified in (V, V/C).
# Temperature is specified in C. Intensity is unitless and scales Isc/Imp.

azur3g30a = solarcell(
    isc=(0.5196, 0.00036),  # short-circuit current, temp coefficient
    voc=(2.690, -0.0062),  # open-circuit voltage, temp coefficient
    imp=(0.5029, 0.00024),  # max-power current, temp coefficient
    vmp=(2.409, -0.0067),  # max-power voltage, temp coefficient
    t=28,  # temperature at which the above parameters are specified
)

array = azur3g30a.array(t=np.full((24, 12), 80), g=np.ones((24, 12)))

print("Isc={:0.2f}A".format(array.isc))  # Isc=6.46A
print("Voc={:0.1f}V".format(array.voc))  # Voc=56.8V
print("Imp={:0.2f}A".format(array.imp))  # Imp=6.26A
print("Vmp={:0.1f}V".format(array.vmp))  # Vmp=49.3V
print("Pmp={:0.0f}W".format(array.pmp))  # Pmp=309W

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

v = np.linspace(0, array.voc, 1000)
ax0.plot(v, array.iv(v)), ax0.grid()
ax1.plot(v, array.pv(v)), ax1.grid()
```

![solarcell example](https://raw.githubusercontent.com/amosborne/solarcell/main/examples/readme.png)

## Background

A numeric optimization procedure is used to best fit the classic photovoltaic cell single diode model equation to the datasheet parameters. The resulting IV-curve metrics are validated against the predicted values and a warning is emitted if the results differ by more than 2%. Combining cells in series/parallel with different IV-curves is done by linear interpolation. Temperatures are rounded to 0.1C and intensities are rounded to 0.01 (unitless).

When computing a curve, the provided temperatures and intensities are generally organized as follows: `cell(t, g)` accepts single values, `string(t, g)` accepts one-dimensional arrays, and `array(t, g)` accepts two-dimensional arrays.

## References

1. Amit Jain, Avinashi Kapoor, Exact analytical solutions of the parameters of real solar cells using Lambert W-function, Solar Energy Materials and Solar Cells, Volume 81, Issue 2, 2004, Pages 269-277, ISSN 0927-0248, https://doi.org/10.1016/j.solmat.2003.11.018.
