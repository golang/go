// +build linux
// +build openssl
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

#include "goopenssl.h"

int
_goboringcrypto_ECDSA_sign(EVP_MD* md, const uint8_t *msg, size_t msgLen, uint8_t *sig, unsigned int *slen, GO_EC_KEY *eckey)
{
    EVP_PKEY *key = _goboringcrypto_EVP_PKEY_new();
    if (!_goboringcrypto_EVP_PKEY_assign_EC_KEY(key, eckey))
        return 0;
    return _goboringcrypto_EVP_sign(md, NULL, msg, msgLen, sig, slen, key);
}

int
_goboringcrypto_ECDSA_verify(EVP_MD* md, const uint8_t *msg, size_t msgLen, const uint8_t *sig, unsigned int slen, GO_EC_KEY *eckey)
{

    EVP_PKEY *key = _goboringcrypto_EVP_PKEY_new();
    if (!_goboringcrypto_EVP_PKEY_assign_EC_KEY(key, eckey))
        return 0;

    return _goboringcrypto_EVP_verify(md, NULL, msg, msgLen, sig, slen, key);
}
