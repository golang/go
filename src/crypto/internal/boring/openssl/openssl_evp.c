// +build linux
// +build openssl
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

#include "goopenssl.h"

int
_goboringcrypto_EVP_sign(EVP_MD* md, EVP_PKEY_CTX *ctx, const uint8_t *msg, size_t msgLen, uint8_t *sig, unsigned int *slen, EVP_PKEY *key) {
    EVP_MD_CTX *mdctx = NULL;
    int ret = 0;

    if (!(mdctx = _goboringcrypto_EVP_MD_CTX_create()))
        goto err;

    if (1 != _goboringcrypto_EVP_DigestSignInit(mdctx, &ctx, md, NULL, key))
        goto err;

    if (1 != _goboringcrypto_EVP_DigestUpdate(mdctx, msg, msgLen))
        goto err;

    /* Obtain the signature length */
    if (1 != _goboringcrypto_EVP_DigestSignFinal(mdctx, NULL, slen))
        goto err;
    /* Obtain the signature */
    if (1 != _goboringcrypto_EVP_DigestSignFinal(mdctx, sig, slen))
        goto err;

    /* Success */
    ret = 1;

err:
    if (mdctx)
        _goboringcrypto_EVP_MD_CTX_free(mdctx);

    return ret;
}

int
_goboringcrypto_EVP_verify(EVP_MD* md, EVP_PKEY_CTX *ctx, const uint8_t *msg, size_t msgLen, const uint8_t *sig, unsigned int slen, EVP_PKEY *key) {
    EVP_MD_CTX *mdctx = NULL;
    int ret = 0;

    if (!(mdctx = _goboringcrypto_EVP_MD_CTX_create()))
        goto err;
    if (1 != _goboringcrypto_EVP_DigestVerifyInit(mdctx, &ctx, md, NULL, key))
        goto err;

    if (1 != _goboringcrypto_EVP_DigestUpdate(mdctx, msg, msgLen))
        goto err;

    if (1 != _goboringcrypto_EVP_DigestVerifyFinal(mdctx, sig, slen)) {
        goto err;
    }

    /* Success */
    ret = 1;

err:
    if (mdctx)
        _goboringcrypto_EVP_MD_CTX_free(mdctx);

    return ret;
}
