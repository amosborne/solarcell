# solarcell
Estimate IV- and PV-curves given photovoltaic cell datasheet parameters. Capable of generating combined curves for solar arrays with temperature and light intensity gradients. Assumes ideal bypass diodes for cells and ideal blocking diodes for strings.

`pip install solarcell`

## Usage

Please see the examples contained within this repository.

```python
import numpy
import solarcell

# Azur Space 3G30A triple-junction solar cells in a 24s12p configuration.
# Isc/Imp are specified in (A, A/C), Voc/Vmp are specified in (V, V/C).
# Temperature is specified in C. Intensity is unitless and scales Isc/Imp.
array = solarcell.array(
    isc=(0.5196, 0.00036),  # short-circuit current, temp coefficient
    voc=(2.690, -0.0062),   # open-circuit voltage, temp coefficient
    imp=(0.5029, 0.00024),  # max-power current, temp coefficient
    vmp=(2.409, -0.0067),   # max-power voltage, temp coefficient
    t=28,   # temperature at which the above parameters are specified
    ns=24,  # number of series cells in a string
    np=12,  # number of parallel strings in an array
)

# Print the IV- and PV-curve metrics at 80C with nominal intensity.
curve = array.curve(t=80, g=1)
print(curve.isc, curve.voc, curve.imp, curve.vmp, curve.pmp)

# Plot the IV- and PV-curves.
v = numpy.linspace(0, curve.voc, 1000)
fig, (ax0, ax1) = plt.subplot(rows=2, sharex=True)
ax0.plot(v, curve.iv(v))
ax1.plot(v, curve.pv(v))
```

A numeric optimization procedure is used to best fit the classic photovoltaic cell diode equation to the datasheet parameters. The resulting IV-curve metrics are validated against the predicted values and a warning is emitted if the results differ by more than 2%. Combining cells in series/parallel with different IV-curves is done by linear interpolation.

When computing a curve, the provided temperature and intensity may be in the form of a single-value (one for the entire array), a one-dimensional array-like of values (one for each string), or a two-dimensional array-like of values (one for each cell, strings by the second dimension). It is possible to put strings of different lengths in parallel by assigning the intensity of excess cells to zero. Requesting a temperature below absolute zero or an intensity less than zero will raise an error.

IV- and PV- curve functions return a single value when provided a single value as an input. If given a one-dimensional array-like as an input, a one-dimensional Numpy array of the same size is returned. Requesting a result for a voltage that is outside of the (0, Voc) range will raise an error.
