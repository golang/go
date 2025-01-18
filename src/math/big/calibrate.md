# Calibration of Algorithm Thresholds

This document describes the approach to calibration of algorithmic thresholds in
`math/big`, implemented in [calibrate_test.go](calibrate_test.go).

Basic operations like multiplication and division have many possible implementations.
Most algorithms that are better asymptotically have overheads that make them
run slower for small inputs. When presented with an operation to run, `math/big`
must decide which algorithm to use.

For example, for small inputs, multiplication using the “grade school algorithm” is fastest.
Given multi-digit x, y and a target z: clear z, and then for each digit y[i], z[i:] += x\*y[i].
That last operation, adding a vector times a digit to another vector (including carrying up
the vector during the multiplication and addition), can be implemented in a tight assembly loop.
The overall speed is O(N\*\*2) where N is the number of digits in x and y (assume they match),
but the tight inner loop performs well for small inputs.

[Karatsuba's algorithm](https://en.wikipedia.org/wiki/Karatsuba_algorithm)
multiplies two N-digit numbers by splitting them in half, computing
three N/2-digit products, and then reconstructing the final product using a few more
additions and subtractions. It runs in O(N\*\*log₂ 3) = O(N\*\*1.58) time.
The grade school loop runs faster for small inputs,
but eventually Karatsuba's smaller asymptotic run time wins.

The multiplication implementation must decide which to use.
Under the assumption that once Karatsuba is faster for some N,
it will be larger for all larger N as well,
the rule is to use Karatsuba's algorithm when the input length N ≥ karatsubaThreshold.

Calibration is the process of determining what karatsubaThreshold should be set to.
It doesn't sound like it should be that hard, but it is:
- Theoretical analysis does not help: the answer depends on the actual machines
and the actual constant factors in the two implementations.
- We are picking a single karatsubaThreshold for all systems,
despite them having different relative execution speeds for the operations
in the two algorithms.
(We could in theory pick different thresholds for different architectures,
but there can still be significant variation within a given architecture.)
- The assumption that there is a single N where
an asymptotically better algorithm becomes faster and stays faster
is not true in general.
- Recursive algorithms like Karatsuba's may have  different optimal
thresholds for different large input sizes.
- Thresholds can interfere. For example, changing the karatsubaThreshold makes
multiplication faster or slower, which in turn affects the best divRecursiveThreshold
(because divisions use multiplication).

The best we can do is measure the performance of the overall multiplication
algorithm across a variety of inputs and thresholds and look for a threshold
that balances all these concerns reasonably well,
setting thresholds in dependency order (for example, multiplication before division).

The code in `calibrate_test.go` does this measurement of a variety of input sizes
and threshold values and prints the timing results as a CSV file.
The code in `calibrate_graph.go` reads the CSV and writes out an SVG file plotting the data.
For example:

	go test -run=Calibrate/KaratsubaMul -timeout=1h -calibrate >kmul.csv
	go run calibrate_graph.go kmul.csv >kmul.svg

Any particular input is sensitive to only a few transitions in threshold.
For example, an input of size 320 recurses on inputs of size 160,
which recurses on inputs of size 80,
which recurses on inputs of size 40,
and so on, until falling below the Karatsuba threshold.
Here is what the timing looks like for an input of size 320,
normalized so that 1.0 is the fastest timing observed:

![KaratsubaThreshold on an Apple M3 Pro, N=320 only](https://swtch.com/math/big/_calibrate/KaratsubaMul/cal.mac320.svg)

For this input, all thresholds from 21 to 40 perform optimally and identically: they all mean “recurse at N=40 but not at N=20”.
From the single input of size N=320, we cannot decide which of these 20 thresholds is best.

Other inputs exercise other decision points. For example, here is the timing for N=240:

![KaratsubaThreshold on an Apple M3 Pro, N=240 only](https://swtch.com/math/big/_calibrate/KaratsubaMul/cal.mac240.svg)

In this case, all the thresholds from 31 to 60 perform optimally and identically, recursing at N=60 but not N=30.

If we combine these two into a single graph and then plot the geometric mean of the two lines in blue,
the optimal range becomes a little clearer:

![KaratsubaThreshold on an Apple M3 Pro](https://swtch.com/math/big/_calibrate/KaratsubaMul/cal.mac240+320.svg)

The actual calibration runs all possible inputs from size N=200 to N=400, in increments of 8,
plotting all 26 lines in a faded gray (note the changed y-axis scale, zooming in near 1.0).

![KaratsubaThreshold on an Apple M3 Pro](https://swtch.com/math/big/_calibrate/KaratsubaMul/cal.mac.svg)

Now the optimal value is clear: the best threshold on this chip, with these algorithmic implementations, is 40.

Unfortunately, other chips are different. Here is an Intel Xeon server chip:

![KaratsubaThreshold on an Apple M3 Pro](https://swtch.com/math/big/_calibrate/KaratsubaMul/cal.c2s16.svg)

On this chip, the best threshold is closer to 60. Luckily, 40 is not a terrible choice either: it is only about 2% slower on average.

The rest of this document presents the timings measured for the `math/big` thresholds on a variety of machines
and justifies the final thresholds. The timings used these machines:

- The `gotip-linux-amd64_c3h88-perf_vs_release` gomote, a Google Cloud c3-high-88 machine using an Intel Xeon Platinum 8481C CPU (Emerald Rapids).
- The `gotip-linux-amd64_c2s16-perf_vs_release` gomote, a Google Cloud c2-standard-16 machine using an Intel Xeon Gold 6253CL CPU (Cascade Lake).
- A home server built with an AMD Ryzen 9 7950X CPU.
- The `gotip-linux-arm64_c4as16-perf_vs_release` gomote, a Google Cloud c4a-standard-16 machine using Google's Axiom Arm CPU.
- An Apple MacBook Pro with an Apple M3 Pro CPU.

In general, we break ties in favor of the newer c3h88 x86 perf gomote, then the c4as16 arm64 perf gomote, and then the others.

## Karatsuba Multiplication

Here are the full results for the Karatsuba multiplication threshold.

![KaratsubaThreshold on an Intel Xeon Platium 8481C](https://swtch.com/math/big/_calibrate/KaratsubaMul/cal.c3h88.svg)
![KaratsubaThreshold on an Intel Xeon Gold 6253CL](https://swtch.com/math/big/_calibrate/KaratsubaMul/cal.c2s16.svg)
![KaratsubaThreshold on an AMD Ryzen 9 7950X](https://swtch.com/math/big/_calibrate/KaratsubaMul/cal.s7.svg)
![KaratsubaThreshold on an Axiom Arm](https://swtch.com/math/big/_calibrate/KaratsubaMul/cal.c4as16.svg)
![KaratsubaThreshold on an Apple M3 Pro](https://swtch.com/math/big/_calibrate/KaratsubaMul/cal.mac.svg)

The majority of systems have optimum thresholds near 40, so we chose karatsubaThreshold = 40.

## Basic Squaring

For squaring a number (`z.Mul(x, x)`), math/big uses grade school multiplication
up to basicSqrThreshold, where it switches to a customized algorithm that is
still quadratic but avoids half the word-by-word multiplies
since the two arguments are identical.
That algorithm's inner loops are not as tight as the grade school multiplication,
so it is slower for small inputs. How small?

Here are the timings:

![BasicSqrThreshold on an Intel Xeon Platium 8481C](https://swtch.com/math/big/_calibrate/BasicSqr/cal.c3h88.svg)
![BasicSqrThreshold on an Intel Xeon Gold 6253CL](https://swtch.com/math/big/_calibrate/BasicSqr/cal.c2s16.svg)
![BasicSqrThreshold on an AMD Ryzen 9 7950X](https://swtch.com/math/big/_calibrate/BasicSqr/cal.s7.svg)
![BasicSqrThreshold on an Axiom Arm](https://swtch.com/math/big/_calibrate/BasicSqr/cal.c4as16.svg)
![BasicSqrThreshold on an Apple M3 Pro](https://swtch.com/math/big/_calibrate/BasicSqr/cal.mac.svg)

These inputs are so small that the calibration times batches of 100 instead of individual operations.
There is no one best threshold, even on a single system, because some of the sizes seem to run
the grade school algorithm faster than others.
For example, on the AMD CPU,
for N=14, basic squaring is 4% faster than basic multiplication,
suggesting the threshold has been crossed,
but for N=16, basic multiplication is 9% faster than basic squaring,
probably because the tight assembly can use larger chunks.

It is unclear why the Axiom Arm timings are so incredibly noisy.

We chose basicSqrThreshold = 12.

## Karatsuba Squaring

Beyond the basic squaring threshold, at some point a customized Karatsuba can take over.
It uses three half-sized squarings instead of three half-sized multiplies.
Here are the timings:

![KaratsubaSqrThreshold on an Intel Xeon Platium 8481C](https://swtch.com/math/big/_calibrate/KaratsubaSqr/cal.c3h88.svg)
![KaratsubaSqrThreshold on an Intel Xeon Gold 6253CL](https://swtch.com/math/big/_calibrate/KaratsubaSqr/cal.c2s16.svg)
![KaratsubaSqrThreshold on an AMD Ryzen 9 7950X](https://swtch.com/math/big/_calibrate/KaratsubaSqr/cal.s7.svg)
![KaratsubaSqrThreshold on an Axiom Arm](https://swtch.com/math/big/_calibrate/KaratsubaSqr/cal.c4as16.svg)
![KaratsubaSqrThreshold on an Apple M3 Pro](https://swtch.com/math/big/_calibrate/KaratsubaSqr/cal.mac.svg)

The majority of chips preferred a lower threshold, around 60-70,
but the older Intel Xeon and the AMD prefer a threshold around 100-120.

We chose karatsubaSqrThreshold = 80, which is within 2% of optimal on all the chips.

## Recursive Division

Division uses a recursive divide-and-conquer algorithm for large inputs,
eventually falling back to a more traditional grade-school whole-input trial-and-error division.
Here are the timings for the threshold between the two:

![DivRecursiveThreshold on an Intel Xeon Platium 8481C](https://swtch.com/math/big/_calibrate/DivRecursive/cal.c3h88.svg)
![DivRecursiveThreshold on an Intel Xeon Gold 6253CL](https://swtch.com/math/big/_calibrate/DivRecursive/cal.c2s16.svg)
![DivRecursiveThreshold on an AMD Ryzen 9 7950X](https://swtch.com/math/big/_calibrate/DivRecursive/cal.s7.svg)
![DivRecursiveThreshold on an Axiom Arm](https://swtch.com/math/big/_calibrate/DivRecursive/cal.c4as16.svg)
![DivRecursiveThreshold on an Apple M3 Pro](https://swtch.com/math/big/_calibrate/DivRecursive/cal.mac.svg)

We chose divRecursiveThreshold = 40.
