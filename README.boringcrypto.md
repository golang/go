# dev.boringcrypto branch

We have been working inside Google on a fork of Go that uses
BoringCrypto (the core of [BoringSSL](https://boringssl.googlesource.com/boringssl/)) for various crypto primitives, in
furtherance of some [work related to FIPS 140-2](http://csrc.nist.gov/groups/STM/cmvp/documents/140-1/140sp/140sp2964.pdf). We have heard that
some external users of Go would be interested in this code as well, so
I intend to create a new branch dev.boringcrypto that will hold
patches to make Go use BoringCrypto.

Unlike typical dev branches, we do not intend any eventual merge of
this code into the master branch. Instead we intend to maintain in
that branch the latest release plus BoringCrypto patches. In this
sense it is a bit like dev.typealias holding go1.8+type alias patches.

To be clear, we are not making any statements or representations about
the suitability of this code in relation to the FIPS 140-2 standard.
Interested users will have to evaluate for themselves whether the code
is useful for their own purposes.
