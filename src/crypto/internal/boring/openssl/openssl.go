// +build linux
// +build openssl
// +build !android
// +build !cmd_go_bootstrap
// +build !msan

package openssl

// #include "goopenssl.h"
// #cgo LDFLAGS: -ldl
import "C"
import (
	"errors"
	"math/big"
	"os"
	"runtime"
	"strings"
	"unsafe"
)

type openssl struct {
	aes
	ecdsa
	hmac
	rsa
	sha
}

func NewOpenSSL() *openssl {
	return &openssl{
		aes:   aes{},
		ecdsa: ecdsa{},
		hmac:  hmac{},
		rsa:   rsa{},
		sha:   sha{},
	}
}

func (_ *openssl) Init() {
	// We lock to an OS thread here in order to set up
	// concurrency support in OpenSSL via OpenSSL_thread_setup
	// below.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Check if we can `dlopen` OpenSSL
	if C._goboringcrypto_DLOPEN_OPENSSL() == C.NULL {
		panic(fail("OpenSSL not found"))
	}

	// Initialize the OpenSSL library.
	C._goboringcrypto_OPENSSL_setup()

	if C._goboringcrypto_OPENSSL_thread_setup() != 1 {
		panic(fail("OpenSSL thread setup failed"))
	}
}

// When this variable is true, the go crypto API will panic when a caller
// tries to use the API in a non-compliant manner.  When this is false, the
// go crytpo API will allow existing go crypto APIs to be used even
// if they aren't FIPS compliant.  However, all the unerlying crypto operations
// will still be done by OpenSSL.
var strictFIPS = false

func panicIfStrictFIPS(msg string) {
	if os.Getenv("GOLANG_STRICT_FIPS") == "1" || strictFIPS {
		panic(fail(msg))
	}
}

func newOpenSSLError(msg string) error {
	var b strings.Builder
	var e C.ulong

	b.WriteString(msg)
	b.WriteString("\nopenssl error(s):\n")

	for {
		e = C._goboringcrypto_internal_ERR_get_error()
		if e == 0 {
			break
		}

		var buf [256]byte

		C._goboringcrypto_internal_ERR_error_string_n(e, base(buf[:]), 256)
		b.Write(buf[:])
		b.WriteByte('\n')
	}

	return errors.New(b.String())
}

type fail string

func (e fail) Error() string { return "openssl: " + string(e) + " failed" }

// base returns the address of the underlying array in b,
// being careful not to panic when b has zero length.
func base(b []byte) *C.uint8_t {
	if len(b) == 0 {
		return nil
	}
	return (*C.uint8_t)(unsafe.Pointer(&b[0]))
}

func bigToBN(x *big.Int) *C.GO_BIGNUM {
	if x == nil {
		return nil
	}
	raw := x.Bytes()
	return C._goboringcrypto_BN_bin2bn(base(raw), C.size_t(len(raw)), nil)
}

func bnToBig(bn *C.GO_BIGNUM) *big.Int {
	raw := make([]byte, C._goboringcrypto_BN_num_bytes(bn))
	n := C._goboringcrypto_BN_bn2bin(bn, base(raw))
	return new(big.Int).SetBytes(raw[:n])
}

func bigToBn(bnp **C.GO_BIGNUM, b *big.Int) bool {
	if *bnp != nil {
		C._goboringcrypto_BN_free(*bnp)
		*bnp = nil
	}
	if b == nil {
		return true
	}
	raw := b.Bytes()
	bn := C._goboringcrypto_BN_bin2bn(base(raw), C.size_t(len(raw)), nil)
	if bn == nil {
		return false
	}
	*bnp = bn
	return true
}
