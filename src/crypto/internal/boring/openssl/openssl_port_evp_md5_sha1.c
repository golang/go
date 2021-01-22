// This file contains a backport of the EVP_md5_sha1 method.

// +build linux
// +build openssl
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

// The following is a partial backport of crypto/evp/m_md5_sha1.c,
// commit cbc8a839959418d8a2c2e3ec6bdf394852c9501e on the
// OpenSSL_1_1_0-stable branch.  The ctrl function has been removed.

#include "goopenssl.h"

#if OPENSSL_VERSION_NUMBER < 0x10100000L
// New in OpenSSL 1.1.
static inline void *
_goboringcrypto_internal_EVP_MD_CTX_md_data(EVP_MD_CTX *ctx)
{
  return ctx->md_data;
}

/*
 * Copyright 2015-2016 The OpenSSL Project Authors. All Rights Reserved.
 *
 * Licensed under the OpenSSL license (the "License").  You may not use
 * this file except in compliance with the License.  You can obtain a copy
 * in the file LICENSE in the source distribution or at
 * https://www.openssl.org/source/license.html
 */

#if !defined(OPENSSL_NO_MD5)

#include <openssl/evp.h>
#include <openssl/objects.h>
#include <openssl/x509.h>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <openssl/rsa.h>

struct md5_sha1_ctx {
  MD5_CTX md5;
  SHA_CTX sha1;
};

static int _goboringcrypto_internal_init(EVP_MD_CTX *ctx)
{
  struct md5_sha1_ctx *mctx = _goboringcrypto_internal_EVP_MD_CTX_md_data(ctx);
  if (!_goboringcrypto_internal_MD5_Init(&mctx->md5))
    return 0;
  return _goboringcrypto_SHA1_Init(&mctx->sha1);
}

static int _goboringcrypto_internal_update(EVP_MD_CTX *ctx, const void *data, size_t count)
{
  struct md5_sha1_ctx *mctx = _goboringcrypto_internal_EVP_MD_CTX_md_data(ctx);
  if (!_goboringcrypto_internal_MD5_Update(&mctx->md5, data, count))
    return 0;
  return _goboringcrypto_SHA1_Update(&mctx->sha1, data, count);
}

static int _goboringcrypto_internal_final(EVP_MD_CTX *ctx, unsigned char *md)
{
  struct md5_sha1_ctx *mctx = _goboringcrypto_internal_EVP_MD_CTX_md_data(ctx);
  if (!_goboringcrypto_internal_MD5_Final(md, &mctx->md5))
    return 0;
  return _goboringcrypto_SHA1_Final(md + MD5_DIGEST_LENGTH, &mctx->sha1);
}

// Change: Removed:
// static int ctrl(EVP_MD_CTX *ctx, int cmd, int mslen, void *ms)

static const EVP_MD md5_sha1_md = {
  NID_md5_sha1,
  NID_md5_sha1,
  MD5_DIGEST_LENGTH + SHA_DIGEST_LENGTH,
  0,
  _goboringcrypto_internal_init,
  _goboringcrypto_internal_update,
  _goboringcrypto_internal_final,
  NULL,
  NULL,
  EVP_PKEY_NULL_method, // Change: inserted
  MD5_CBLOCK,
  sizeof(EVP_MD *) + sizeof(struct md5_sha1_ctx),
  NULL, // Change: was ctrl
};

// Change: Apply name mangling.
const GO_EVP_MD* _goboringcrypto_backport_EVP_md5_sha1(void) {
  return &md5_sha1_md;
}
#endif
#endif

