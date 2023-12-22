# solarcell
Estimate IV- and PV-curves given photovoltaic cell datasheet parameters. Capable of generating combined curves for solar arrays with temperature and light intensity gradients. Assumes ideal bypass and blocking diodes for all cells.

`pip install solarcell`

## Usage

Please see the examples contained within this repository.

```python
import solarcell
import numpy
from matplotlib import pyplot as plt

# Azur Space 3G30A triple-junction solar cells in a 24s12p configuration.
# Isc/Imp are specified in (A, A/C), Voc/Vmp are specified in (V, V/C).
# Temperature is specified in C. Intensity is unitless and scales Isc/Imp.

array = solarcell.array(
    isc=(0.5196, 0.00036),  # short-circuit current, temp coefficient
    voc=(2.690, -0.0062),  # open-circuit voltage, temp coefficient
    imp=(0.5029, 0.00024),  # max-power current, temp coefficient
    vmp=(2.409, -0.0067),  # max-power voltage, temp coefficient
    t=28,  # temperature at which the above parameters are specified
    ns=24,  # number of series cells in a string
    np=12,  # number of parallel strings in an array
)

curve = array.curve(t=80, g=1)

print("Isc={:0.2f}A".format(curve.isc))  # Isc=6.46A
print("Voc={:0.1f}V".format(curve.voc))  # Voc=56.8V
print("Imp={:0.2f}A".format(curve.imp))  # Imp=6.26A
print("Vmp={:0.1f}V".format(curve.vmp))  # Vmp=49.3V
print("Pmp={:0.0f}W".format(curve.pmp))  # Pmp=309W

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

v = numpy.linspace(0, curve.voc, 1000)
ax0.plot(v, curve.iv(v)), ax0.grid()
ax1.plot(v, curve.pv(v)), ax1.grid()
```

![solarcell example](https://github.com/amosborne/solarcell/readme.png)

## Background

A numeric optimization procedure is used to best fit the classic photovoltaic cell single diode model equation to the datasheet parameters. The resulting IV-curve metrics are validated against the predicted values and a warning is emitted if the results differ by more than 2%. Combining cells in series/parallel with different IV-curves is done by linear interpolation.

When computing a curve, the provided temperature and intensity may be in the form of a single-value (one for the entire array), a one-dimensional array-like of values (one for each string), or a two-dimensional array-like of values (one for each cell, strings by the second dimension). It is possible to put strings of different lengths in parallel by assigning the intensity of excess cells to zero.

## References

1. Amit Jain, Avinashi Kapoor, Exact analytical solutions of the parameters of real solar cells using Lambert W-function, Solar Energy Materials and Solar Cells, Volume 81, Issue 2, 2004, Pages 269-277, ISSN 0927-0248, https://doi.org/10.1016/j.solmat.2003.11.018.
