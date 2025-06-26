# Unified Test Suite for Go-Panikint

This directory contains comprehensive unit tests for the Go-Panikint compiler's overflow and truncation detection features.

## Test Files

- **arithmetic_test.go**: Tests for arithmetic overflow detection (addition, subtraction, multiplication, division)
- **truncation_test.go**: Tests for integer type truncation detection
- **comprehensive_test.go**: Edge cases, fuzzing tests, and real-world scenarios

## Running Tests

```bash
# Run all tests
GOROOT=/path/to/go-panikint /path/to/go-panikint/bin/go test -v

# Run specific test file
GOROOT=/path/to/go-panikint /path/to/go-panikint/bin/go test -v -run TestSignedInt8Overflow

# Run fuzzing tests
GOROOT=/path/to/go-panikint /path/to/go-panikint/bin/go test -fuzz=FuzzIntegerOverflow -v

# Disable overflow detection
IDC_ABOUT_OVERFLOW=true GOROOT=/path/to/go-panikint /path/to/go-panikint/bin/go test -v

# Disable truncation detection
IDC_ABOUT_TRUNCATION=true GOROOT=/path/to/go-panikint /path/to/go-panikint/bin/go test -v
```

## Test Coverage

### Arithmetic Operations
- **Signed integers**: int8, int16, int32 overflow/underflow
- **Unsigned integers**: uint8, uint16, uint32, uint64 overflow/underflow
- **Operations**: Addition (+), Subtraction (-), Multiplication (*), Division (/)
- **Special cases**: MIN_INT / -1 overflow, division by zero

### Type Truncation
- **Signed conversions**: int64→int32, int32→int16, int16→int8
- **Unsigned conversions**: uint64→uint32, uint32→uint16, uint16→uint8
- **Mixed conversions**: signed to unsigned with negative values
- **Platform-dependent**: int→int32/int16/int8, uint→uint32/uint16/uint8

### Edge Cases
- Maximum and minimum value operations
- Complex truncation chains
- Runtime computed values
- Real-world security scenarios (buffer sizes, array indices)

All tests use proper Go unit testing conventions with `testing.T` and panic recovery patterns.