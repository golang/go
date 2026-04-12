[SignASN1], [Sign], and [PrivateKey.Sign] now return an error, and [VerifyASN1]
and [Verify] now return false, if the hash is shorter than 16 bytes or longer
than 64 bytes. Such inputs are not the output of a supported hash function, and
suggest a message is being passed instead of its hash, which can be severely
insecure.
