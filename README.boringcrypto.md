# dev.boringcrypto branch

We have been working inside Google on a fork of Go that uses
BoringCrypto (the core of [BoringSSL][]) for various crypto
primitives, in furtherance of some [work related to FIPS 140-2][sp].
We have heard that some external users of Go would be interested in
this code as well, so this branch holds the patches to make Go use
BoringCrypto.

[BoringSSL]: https://boringssl.googlesource.com/boringssl/
[sp]: https://csrc.nist.gov/CSRC/media/projects/cryptographic-module-validation-program/documents/security-policies/140sp3678.pdf

Unlike typical dev branches, we do not intend any eventual merge of
this code into the master branch. Instead we intend to maintain in
this branch the latest release plus BoringCrypto patches.

To be clear, we are not making any statements or representations about
the suitability of this code in relation to the FIPS 140-2 standard.
Interested users will have to evaluate for themselves whether the code
is useful for their own purposes.
