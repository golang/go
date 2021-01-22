// +build linux
// +build openssl
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

#include "goopenssl.h"

void
_goboringcrypto_EVP_AES_ctr128_enc(EVP_CIPHER_CTX *ctx, const uint8_t* in, uint8_t* out, size_t in_len)
{
	int len;
	_goboringcrypto_EVP_EncryptUpdate(ctx, out, &len, in, in_len);
}
