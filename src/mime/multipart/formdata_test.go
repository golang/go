// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"net/textproto"
	"os"
	"strings"
	"testing"
)

func TestReadForm(t *testing.T) {
	b := strings.NewReader(strings.ReplaceAll(message, "\n", "\r\n"))
	r := NewReader(b, boundary)
	f, err := r.ReadForm(25)
	if err != nil {
		t.Fatal("ReadForm:", err)
	}
	defer f.RemoveAll()
	if g, e := f.Value["texta"][0], textaValue; g != e {
		t.Errorf("texta value = %q, want %q", g, e)
	}
	if g, e := f.Value["textb"][0], textbValue; g != e {
		t.Errorf("texta value = %q, want %q", g, e)
	}
	fd := testFile(t, f.File["filea"][0], "filea.txt", fileaContents)
	if _, ok := fd.(*os.File); ok {
		t.Error("file is *os.File, should not be")
	}
	fd.Close()
	fd = testFile(t, f.File["fileb"][0], "fileb.txt", filebContents)
	if _, ok := fd.(*os.File); !ok {
		t.Errorf("file has unexpected underlying type %T", fd)
	}
	fd.Close()
}

func TestReadFormWithNamelessFile(t *testing.T) {
	b := strings.NewReader(strings.ReplaceAll(messageWithFileWithoutName, "\n", "\r\n"))
	r := NewReader(b, boundary)
	f, err := r.ReadForm(25)
	if err != nil {
		t.Fatal("ReadForm:", err)
	}
	defer f.RemoveAll()

	if g, e := f.Value["hiddenfile"][0], filebContents; g != e {
		t.Errorf("hiddenfile value = %q, want %q", g, e)
	}
}

// Issue 58384: Handle ReadForm(math.MaxInt64)
func TestReadFormWitFileNameMaxMemoryOverflow(t *testing.T) {
	b := strings.NewReader(strings.ReplaceAll(messageWithFileName, "\n", "\r\n"))
	r := NewReader(b, boundary)
	f, err := r.ReadForm(math.MaxInt64)
	if err != nil {
		t.Fatalf("ReadForm(MaxInt64): %v", err)
	}
	defer f.RemoveAll()

	fd := testFile(t, f.File["filea"][0], "filea.txt", fileaContents)
	if _, ok := fd.(*os.File); ok {
		t.Error("file is *os.File, should not be")
	}
	fd.Close()
}

// Issue 40430: Handle ReadForm(math.MaxInt64)
func TestReadFormMaxMemoryOverflow(t *testing.T) {
	b := strings.NewReader(strings.ReplaceAll(messageWithTextContentType, "\n", "\r\n"))
	r := NewReader(b, boundary)
	f, err := r.ReadForm(math.MaxInt64)
	if err != nil {
		t.Fatalf("ReadForm(MaxInt64): %v", err)
	}
	if f == nil {
		t.Fatal("ReadForm(MaxInt64): missing form")
	}
	defer f.RemoveAll()

	if g, e := f.Value["texta"][0], textaValue; g != e {
		t.Errorf("texta value = %q, want %q", g, e)
	}
}

func TestReadFormWithTextContentType(t *testing.T) {
	// From https://github.com/golang/go/issues/24041
	b := strings.NewReader(strings.ReplaceAll(messageWithTextContentType, "\n", "\r\n"))
	r := NewReader(b, boundary)
	f, err := r.ReadForm(25)
	if err != nil {
		t.Fatal("ReadForm:", err)
	}
	defer f.RemoveAll()

	if g, e := f.Value["texta"][0], textaValue; g != e {
		t.Errorf("texta value = %q, want %q", g, e)
	}
}

func testFile(t *testing.T, fh *FileHeader, efn, econtent string) File {
	if fh.Filename != efn {
		t.Errorf("filename = %q, want %q", fh.Filename, efn)
	}
	if fh.Size != int64(len(econtent)) {
		t.Errorf("size = %d, want %d", fh.Size, len(econtent))
	}
	f, err := fh.Open()
	if err != nil {
		t.Fatal("opening file:", err)
	}
	b := new(strings.Builder)
	_, err = io.Copy(b, f)
	if err != nil {
		t.Fatal("copying contents:", err)
	}
	if g := b.String(); g != econtent {
		t.Errorf("contents = %q, want %q", g, econtent)
	}
	return f
}

const (
	fileaContents = "This is a test file."
	filebContents = "Another test file."
	textaValue    = "foo"
	textbValue    = "bar"
	boundary      = `MyBoundary`
)

const messageWithFileWithoutName = `
--MyBoundary
Content-Disposition: form-data; name="hiddenfile"; filename=""
Content-Type: text/plain

` + filebContents + `
--MyBoundary--
`

const messageWithFileName = `
--MyBoundary
Content-Disposition: form-data; name="filea"; filename="filea.txt"
Content-Type: text/plain

` + fileaContents + `
--MyBoundary--
`

const messageWithTextContentType = `
--MyBoundary
Content-Disposition: form-data; name="texta"
Content-Type: text/plain

` + textaValue + `
--MyBoundary
`

const message = `
--MyBoundary
Content-Disposition: form-data; name="filea"; filename="filea.txt"
Content-Type: text/plain

` + fileaContents + `
--MyBoundary
Content-Disposition: form-data; name="fileb"; filename="fileb.txt"
Content-Type: text/plain

` + filebContents + `
--MyBoundary
Content-Disposition: form-data; name="texta"

` + textaValue + `
--MyBoundary
Content-Disposition: form-data; name="textb"

` + textbValue + `
--MyBoundary--
`

func TestReadForm_NoReadAfterEOF(t *testing.T) {
	maxMemory := int64(32) << 20
	boundary := `---------------------------8d345eef0d38dc9`
	body := `
-----------------------------8d345eef0d38dc9
Content-Disposition: form-data; name="version"

171
-----------------------------8d345eef0d38dc9--`

	mr := NewReader(&failOnReadAfterErrorReader{t: t, r: strings.NewReader(body)}, boundary)

	f, err := mr.ReadForm(maxMemory)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Got: %#v", f)
}

// failOnReadAfterErrorReader is an io.Reader wrapping r.
// It fails t if any Read is called after a failing Read.
type failOnReadAfterErrorReader struct {
	t      *testing.T
	r      io.Reader
	sawErr error
}

func (r *failOnReadAfterErrorReader) Read(p []byte) (n int, err error) {
	if r.sawErr != nil {
		r.t.Fatalf("unexpected Read on Reader after previous read saw error %v", r.sawErr)
	}
	n, err = r.r.Read(p)
	r.sawErr = err
	return
}

// TestReadForm_NonFileMaxMemory asserts that the ReadForm maxMemory limit is applied
// while processing non-file form data as well as file form data.
func TestReadForm_NonFileMaxMemory(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in -short mode")
	}
	n := 10 << 20
	largeTextValue := strings.Repeat("1", n)
	message := `--MyBoundary
Content-Disposition: form-data; name="largetext"

` + largeTextValue + `
--MyBoundary--
`
	testBody := strings.ReplaceAll(message, "\n", "\r\n")
	// Try parsing the form with increasing maxMemory values.
	// Changes in how we account for non-file form data may cause the exact point
	// where we change from rejecting the form as too large to accepting it to vary,
	// but we should see both successes and failures.
	const failWhenMaxMemoryLessThan = 128
	for maxMemory := int64(0); maxMemory < failWhenMaxMemoryLessThan*2; maxMemory += 16 {
		b := strings.NewReader(testBody)
		r := NewReader(b, boundary)
		f, err := r.ReadForm(maxMemory)
		if err != nil {
			continue
		}
		if g := f.Value["largetext"][0]; g != largeTextValue {
			t.Errorf("largetext mismatch: got size: %v, expected size: %v", len(g), len(largeTextValue))
		}
		f.RemoveAll()
		if maxMemory < failWhenMaxMemoryLessThan {
			t.Errorf("ReadForm(%v): no error, expect to hit memory limit when maxMemory < %v", maxMemory, failWhenMaxMemoryLessThan)
		}
		return
	}
	t.Errorf("ReadForm(x) failed for x < 1024, expect success")
}

// TestReadForm_MetadataTooLarge verifies that we account for the size of field names,
// MIME headers, and map entry overhead while limiting the memory consumption of parsed forms.
func TestReadForm_MetadataTooLarge(t *testing.T) {
	for _, test := range []struct {
		name string
		f    func(*Writer)
	}{{
		name: "large name",
		f: func(fw *Writer) {
			name := strings.Repeat("a", 10<<20)
			w, _ := fw.CreateFormField(name)
			w.Write([]byte("value"))
		},
	}, {
		name: "large MIME header",
		f: func(fw *Writer) {
			h := make(textproto.MIMEHeader)
			h.Set("Content-Disposition", `form-data; name="a"`)
			h.Set("X-Foo", strings.Repeat("a", 10<<20))
			w, _ := fw.CreatePart(h)
			w.Write([]byte("value"))
		},
	}, {
		name: "many parts",
		f: func(fw *Writer) {
			for i := 0; i < 110000; i++ {
				w, _ := fw.CreateFormField("f")
				w.Write([]byte("v"))
			}
		},
	}} {
		t.Run(test.name, func(t *testing.T) {
			var buf bytes.Buffer
			fw := NewWriter(&buf)
			test.f(fw)
			if err := fw.Close(); err != nil {
				t.Fatal(err)
			}
			fr := NewReader(&buf, fw.Boundary())
			_, err := fr.ReadForm(0)
			if err != ErrMessageTooLarge {
				t.Errorf("fr.ReadForm() = %v, want ErrMessageTooLarge", err)
			}
		})
	}
}

// TestReadForm_ManyFiles_Combined tests that a multipart form containing many files only
// results in a single on-disk file.
func TestReadForm_ManyFiles_Combined(t *testing.T) {
	const distinct = false
	testReadFormManyFiles(t, distinct)
}

// TestReadForm_ManyFiles_Distinct tests that setting GODEBUG=multipartfiles=distinct
// results in every file in a multipart form being placed in a distinct on-disk file.
func TestReadForm_ManyFiles_Distinct(t *testing.T) {
	t.Setenv("GODEBUG", "multipartfiles=distinct")
	const distinct = true
	testReadFormManyFiles(t, distinct)
}

func testReadFormManyFiles(t *testing.T, distinct bool) {
	var buf bytes.Buffer
	fw := NewWriter(&buf)
	const numFiles = 10
	for i := 0; i < numFiles; i++ {
		name := fmt.Sprint(i)
		w, err := fw.CreateFormFile(name, name)
		if err != nil {
			t.Fatal(err)
		}
		w.Write([]byte(name))
	}
	if err := fw.Close(); err != nil {
		t.Fatal(err)
	}
	fr := NewReader(&buf, fw.Boundary())
	fr.tempDir = t.TempDir()
	form, err := fr.ReadForm(0)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < numFiles; i++ {
		name := fmt.Sprint(i)
		if got := len(form.File[name]); got != 1 {
			t.Fatalf("form.File[%q] has %v entries, want 1", name, got)
		}
		fh := form.File[name][0]
		file, err := fh.Open()
		if err != nil {
			t.Fatalf("form.File[%q].Open() = %v", name, err)
		}
		if distinct {
			if _, ok := file.(*os.File); !ok {
				t.Fatalf("form.File[%q].Open: %T, want *os.File", name, file)
			}
		}
		got, err := io.ReadAll(file)
		file.Close()
		if string(got) != name || err != nil {
			t.Fatalf("read form.File[%q]: %q, %v; want %q, nil", name, string(got), err, name)
		}
	}
	dir, err := os.Open(fr.tempDir)
	if err != nil {
		t.Fatal(err)
	}
	defer dir.Close()
	names, err := dir.Readdirnames(0)
	if err != nil {
		t.Fatal(err)
	}
	wantNames := 1
	if distinct {
		wantNames = numFiles
	}
	if len(names) != wantNames {
		t.Fatalf("temp dir contains %v files; want 1", len(names))
	}
	if err := form.RemoveAll(); err != nil {
		t.Fatalf("form.RemoveAll() = %v", err)
	}
	names, err = dir.Readdirnames(0)
	if err != nil {
		t.Fatal(err)
	}
	if len(names) != 0 {
		t.Fatalf("temp dir contains %v files; want 0", len(names))
	}
}

func TestReadFormLimits(t *testing.T) {
	for _, test := range []struct {
		values           int
		files            int
		extraKeysPerFile int
		wantErr          error
		godebug          string
	}{
		{values: 1000},
		{values: 1001, wantErr: ErrMessageTooLarge},
		{values: 500, files: 500},
		{values: 501, files: 500, wantErr: ErrMessageTooLarge},
		{files: 1000},
		{files: 1001, wantErr: ErrMessageTooLarge},
		{files: 1, extraKeysPerFile: 9998}, // plus Content-Disposition and Content-Type
		{files: 1, extraKeysPerFile: 10000, wantErr: ErrMessageTooLarge},
		{godebug: "multipartmaxparts=100", values: 100},
		{godebug: "multipartmaxparts=100", values: 101, wantErr: ErrMessageTooLarge},
		{godebug: "multipartmaxheaders=100", files: 2, extraKeysPerFile: 48},
		{godebug: "multipartmaxheaders=100", files: 2, extraKeysPerFile: 50, wantErr: ErrMessageTooLarge},
	} {
		name := fmt.Sprintf("values=%v/files=%v/extraKeysPerFile=%v", test.values, test.files, test.extraKeysPerFile)
		if test.godebug != "" {
			name += fmt.Sprintf("/godebug=%v", test.godebug)
		}
		t.Run(name, func(t *testing.T) {
			if test.godebug != "" {
				t.Setenv("GODEBUG", test.godebug)
			}
			var buf bytes.Buffer
			fw := NewWriter(&buf)
			for i := 0; i < test.values; i++ {
				w, _ := fw.CreateFormField(fmt.Sprintf("field%v", i))
				fmt.Fprintf(w, "value %v", i)
			}
			for i := 0; i < test.files; i++ {
				h := make(textproto.MIMEHeader)
				h.Set("Content-Disposition",
					fmt.Sprintf(`form-data; name="file%v"; filename="file%v"`, i, i))
				h.Set("Content-Type", "application/octet-stream")
				for j := 0; j < test.extraKeysPerFile; j++ {
					h.Set(fmt.Sprintf("k%v", j), "v")
				}
				w, _ := fw.CreatePart(h)
				fmt.Fprintf(w, "value %v", i)
			}
			if err := fw.Close(); err != nil {
				t.Fatal(err)
			}
			fr := NewReader(bytes.NewReader(buf.Bytes()), fw.Boundary())
			form, err := fr.ReadForm(1 << 10)
			if err == nil {
				defer form.RemoveAll()
			}
			if err != test.wantErr {
				t.Errorf("ReadForm = %v, want %v", err, test.wantErr)
			}
		})
	}
}

func TestReadFormEndlessHeaderLine(t *testing.T) {
	for _, test := range []struct {
		name   string
		prefix string
	}{{
		name:   "name",
		prefix: "X-",
	}, {
		name:   "value",
		prefix: "X-Header: ",
	}, {
		name:   "continuation",
		prefix: "X-Header: foo\r\n  ",
	}} {
		t.Run(test.name, func(t *testing.T) {
			const eol = "\r\n"
			s := `--boundary` + eol
			s += `Content-Disposition: form-data; name="a"` + eol
			s += `Content-Type: text/plain` + eol
			s += test.prefix
			fr := io.MultiReader(
				strings.NewReader(s),
				neverendingReader('X'),
			)
			r := NewReader(fr, "boundary")
			_, err := r.ReadForm(1 << 20)
			if err != ErrMessageTooLarge {
				t.Fatalf("ReadForm(1 << 20): %v, want ErrMessageTooLarge", err)
			}
		})
	}
}

type neverendingReader byte

func (r neverendingReader) Read(p []byte) (n int, err error) {
	for i := range p {
		p[i] = byte(r)
	}
	return len(p), nil
}

func BenchmarkReadForm(b *testing.B) {
	for _, test := range []struct {
		name string
		form func(fw *Writer, count int)
	}{{
		name: "fields",
		form: func(fw *Writer, count int) {
			for i := 0; i < count; i++ {
				w, _ := fw.CreateFormField(fmt.Sprintf("field%v", i))
				fmt.Fprintf(w, "value %v", i)
			}
		},
	}, {
		name: "files",
		form: func(fw *Writer, count int) {
			for i := 0; i < count; i++ {
				w, _ := fw.CreateFormFile(fmt.Sprintf("field%v", i), fmt.Sprintf("file%v", i))
				fmt.Fprintf(w, "value %v", i)
			}
		},
	}} {
		b.Run(test.name, func(b *testing.B) {
			for _, maxMemory := range []int64{
				0,
				1 << 20,
			} {
				var buf bytes.Buffer
				fw := NewWriter(&buf)
				test.form(fw, 10)
				if err := fw.Close(); err != nil {
					b.Fatal(err)
				}
				b.Run(fmt.Sprintf("maxMemory=%v", maxMemory), func(b *testing.B) {
					b.ReportAllocs()
					for i := 0; i < b.N; i++ {
						fr := NewReader(bytes.NewReader(buf.Bytes()), fw.Boundary())
						form, err := fr.ReadForm(maxMemory)
						if err != nil {
							b.Fatal(err)
						}
						form.RemoveAll()
					}

				})
			}
		})
	}
}
