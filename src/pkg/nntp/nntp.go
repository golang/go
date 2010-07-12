// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The nntp package implements a client for the news protocol NNTP,
// as defined in RFC 3977.
package nntp

import (
	"bufio"
	"bytes"
	"container/vector"
	"fmt"
	"http"
	"io"
	"io/ioutil"
	"os"
	"net"
	"sort"
	"strconv"
	"strings"
	"time"
)

// timeFormatNew is the NNTP time format string for NEWNEWS / NEWGROUPS
const timeFormatNew = "20060102 150405"

// timeFormatDate is the NNTP time format string for responses to the DATE command
const timeFormatDate = "20060102150405"

// An Error represents an error response from an NNTP server.
type Error struct {
	Code uint
	Msg  string
}

// A ProtocolError represents responses from an NNTP server
// that seem incorrect for NNTP.
type ProtocolError string

// A Conn represents a connection to an NNTP server. The connection with
// an NNTP server is stateful; it keeps track of what group you have
// selected, if any, and (if you have a group selected) which article is
// current, next, or previous.
//
// Some methods that return information about a specific message take
// either a message-id, which is global across all NNTP servers, groups,
// and messages, or a message-number, which is an integer number that is
// local to the NNTP session and currently selected group.
//
// For all methods that return an io.Reader (or an *Article, which contains
// an io.Reader), that io.Reader is only valid until the next call to a
// method of Conn.
type Conn struct {
	conn  io.WriteCloser
	r     *bufio.Reader
	br    *bodyReader
	close bool
}

// A Group gives information about a single news group on the server.
type Group struct {
	Name string
	// High and low message-numbers
	High, Low int
	// Status indicates if general posting is allowed --
	// typical values are "y", "n", or "m".
	Status string
}

// An Article represents an NNTP article.
type Article struct {
	Header map[string][]string
	Body   io.Reader
}

// A bodyReader satisfies reads by reading from the connection
// until it finds a line containing just .
type bodyReader struct {
	c   *Conn
	eof bool
	buf *bytes.Buffer
}

var dotnl = []byte(".\n")
var dotdot = []byte("..")

func (r *bodyReader) Read(p []byte) (n int, err os.Error) {
	if r.eof {
		return 0, os.EOF
	}
	if r.buf == nil {
		r.buf = &bytes.Buffer{}
	}
	if r.buf.Len() == 0 {
		b, err := r.c.r.ReadBytes('\n')
		if err != nil {
			return 0, err
		}
		// canonicalize newlines
		if b[len(b)-2] == '\r' { // crlf->lf
			b = b[0 : len(b)-1]
			b[len(b)-1] = '\n'
		}
		// stop on .
		if bytes.Equal(b, dotnl) {
			r.eof = true
			return 0, os.EOF
		}
		// unescape leading ..
		if bytes.HasPrefix(b, dotdot) {
			b = b[1:]
		}
		r.buf.Write(b)
	}
	n, _ = r.buf.Read(p)
	return
}

func (r *bodyReader) discard() os.Error {
	_, err := ioutil.ReadAll(r)
	return err
}

// articleReader satisfies reads by dumping out an article's headers
// and body.
type articleReader struct {
	a          *Article
	headerdone bool
	headerbuf  *bytes.Buffer
}

func (r *articleReader) Read(p []byte) (n int, err os.Error) {
	if r.headerbuf == nil {
		buf := new(bytes.Buffer)
		for k, fv := range r.a.Header {
			for _, v := range fv {
				fmt.Fprintf(buf, "%s: %s\n", k, v)
			}
		}
		if r.a.Body != nil {
			fmt.Fprintf(buf, "\n")
		}
		r.headerbuf = buf
	}
	if !r.headerdone {
		n, err = r.headerbuf.Read(p)
		if err == os.EOF {
			err = nil
			r.headerdone = true
		}
		if n > 0 {
			return
		}
	}
	if r.a.Body != nil {
		n, err = r.a.Body.Read(p)
		if err == os.EOF {
			r.a.Body = nil
		}
		return
	}
	return 0, os.EOF
}

func (a *Article) String() string {
	id, ok := a.Header["Message-Id"]
	if !ok {
		return "[NNTP article]"
	}
	return fmt.Sprintf("[NNTP article %s]", id[0])
}

func (a *Article) WriteTo(w io.Writer) (int64, os.Error) {
	return io.Copy(w, &articleReader{a: a})
}

func (p ProtocolError) String() string {
	return string(p)
}

func (e Error) String() string {
	return fmt.Sprintf("%03d %s", e.Code, e.Msg)
}

func maybeId(cmd, id string) string {
	if len(id) > 0 {
		return cmd + " " + id
	}
	return cmd
}

// Dial connects to an NNTP server.
// The network and addr are passed to net.Dial to
// make the connection.
//
// Example:
//   conn, err := nntp.Dial("tcp", "my.news:nntp")
//
func Dial(network, addr string) (*Conn, os.Error) {
	res := new(Conn)
	c, err := net.Dial(network, "", addr)
	if err != nil {
		return nil, err
	}

	res.conn = c
	if res.r, err = bufio.NewReaderSize(c, 4096); err != nil {
		return nil, err
	}

	_, err = res.r.ReadString('\n')
	if err != nil {
		return nil, err
	}

	return res, nil
}

func (c *Conn) body() io.Reader {
	c.br = &bodyReader{c: c}
	return c.br
}

// readStrings reads a list of strings from the NNTP connection,
// stopping at a line containing only a . (Convenience method for
// LIST, etc.)
func (c *Conn) readStrings() ([]string, os.Error) {
	var sv vector.StringVector
	for {
		line, err := c.r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		if strings.HasSuffix(line, "\r\n") {
			line = line[0 : len(line)-2]
		} else if strings.HasSuffix(line, "\n") {
			line = line[0 : len(line)-1]
		}
		if line == "." {
			break
		}
		sv.Push(line)
	}
	return []string(sv), nil
}

// Authenticate logs in to the NNTP server.
// It only sends the password if the server requires one.
func (c *Conn) Authenticate(username, password string) os.Error {
	code, _, err := c.cmd(2, "AUTHINFO USER %s", username)
	if code/100 == 3 {
		_, _, err = c.cmd(2, "AUTHINFO PASS %s", password)
	}
	return err
}

// cmd executes an NNTP command:
// It sends the command given by the format and arguments, and then
// reads the response line. If expectCode > 0, the status code on the
// response line must match it. 1 digit expectCodes only check the first
// digit of the status code, etc.
func (c *Conn) cmd(expectCode uint, format string, args ...interface{}) (code uint, line string, err os.Error) {
	if c.close {
		return 0, "", ProtocolError("connection closed")
	}
	if c.br != nil {
		if err := c.br.discard(); err != nil {
			return 0, "", err
		}
		c.br = nil
	}
	if _, err := fmt.Fprintf(c.conn, format+"\r\n", args); err != nil {
		return 0, "", err
	}
	line, err = c.r.ReadString('\n')
	if err != nil {
		return 0, "", err
	}
	line = strings.TrimSpace(line)
	if len(line) < 4 || line[3] != ' ' {
		return 0, "", ProtocolError("short response: " + line)
	}
	code, err = strconv.Atoui(line[0:3])
	if err != nil {
		return 0, "", ProtocolError("invalid response code: " + line)
	}
	line = line[4:]
	if 1 <= expectCode && expectCode < 10 && code/100 != expectCode ||
		10 <= expectCode && expectCode < 100 && code/10 != expectCode ||
		100 <= expectCode && expectCode < 1000 && code != expectCode {
		err = Error{code, line}
	}
	return
}

// ModeReader switches the NNTP server to "reader" mode, if it
// is a mode-switching server.
func (c *Conn) ModeReader() os.Error {
	_, _, err := c.cmd(20, "MODE READER")
	return err
}

// NewGroups returns a list of groups added since the given time.
func (c *Conn) NewGroups(since *time.Time) ([]Group, os.Error) {
	if _, _, err := c.cmd(231, "NEWGROUPS %s GMT", since.Format(timeFormatNew)); err != nil {
		return nil, err
	}
	return c.readGroups()
}

func (c *Conn) readGroups() ([]Group, os.Error) {
	lines, err := c.readStrings()
	if err != nil {
		return nil, err
	}
	return parseGroups(lines)
}

// NewNews returns a list of the IDs of articles posted
// to the given group since the given time.
func (c *Conn) NewNews(group string, since *time.Time) ([]string, os.Error) {
	if _, _, err := c.cmd(230, "NEWNEWS %s %s GMT", group, since.Format(timeFormatNew)); err != nil {
		return nil, err
	}

	id, err := c.readStrings()
	if err != nil {
		return nil, err
	}

	sort.SortStrings(id)
	w := 0
	for r, s := range id {
		if r == 0 || id[r-1] != s {
			id[w] = s
			w++
		}
	}
	id = id[0:w]

	return id, nil
}

// parseGroups is used to parse a list of group states.
func parseGroups(lines []string) ([]Group, os.Error) {
	var res vector.Vector
	for _, line := range lines {
		ss := strings.Split(strings.TrimSpace(line), " ", 4)
		if len(ss) < 4 {
			return nil, ProtocolError("short group info line: " + line)
		}
		high, err := strconv.Atoi(ss[1])
		if err != nil {
			return nil, ProtocolError("bad number in line: " + line)
		}
		low, err := strconv.Atoi(ss[2])
		if err != nil {
			return nil, ProtocolError("bad number in line: " + line)
		}
		res.Push(&Group{ss[0], high, low, ss[3]})
	}
	realres := make([]Group, res.Len())
	for i, v := range res {
		realres[i] = *v.(*Group)
	}
	return realres, nil
}

// Capabilities returns a list of features this server performs.
// Not all servers support capabilities.
func (c *Conn) Capabilities() ([]string, os.Error) {
	if _, _, err := c.cmd(101, "CAPABILITIES"); err != nil {
		return nil, err
	}
	return c.readStrings()
}

// Date returns the current time on the server.
// Typically the time is later passed to NewGroups or NewNews.
func (c *Conn) Date() (*time.Time, os.Error) {
	_, line, err := c.cmd(111, "DATE")
	if err != nil {
		return nil, err
	}
	t, err := time.Parse(timeFormatDate, line)
	if err != nil {
		return nil, ProtocolError("invalid time: " + line)
	}
	return t, nil
}

// List returns a list of groups present on the server.
// Valid forms are:
//
//   List() - return active groups
//   List(keyword) - return different kinds of information about groups
//   List(keyword, pattern) - filter groups against a glob-like pattern called a wildmat
//
func (c *Conn) List(a ...string) ([]string, os.Error) {
	if len(a) > 2 {
		return nil, ProtocolError("List only takes up to 2 arguments")
	}
	cmd := "LIST"
	if len(a) > 0 {
		cmd += " " + a[0]
		if len(a) > 1 {
			cmd += " " + a[1]
		}
	}
	if _, _, err := c.cmd(215, cmd); err != nil {
		return nil, err
	}
	return c.readStrings()
}

// Group changes the current group.
func (c *Conn) Group(group string) (number, low, high int, err os.Error) {
	_, line, err := c.cmd(211, "GROUP %s", group)
	if err != nil {
		return
	}

	ss := strings.Split(line, " ", 4) // intentional -- we ignore optional message
	if len(ss) < 3 {
		err = ProtocolError("bad group response: " + line)
		return
	}

	var n [3]int
	for i, _ := range n {
		c, err := strconv.Atoi(ss[i])
		if err != nil {
			err = ProtocolError("bad group response: " + line)
			return
		}
		n[i] = c
	}
	number, low, high = n[0], n[1], n[2]
	return
}

// Help returns the server's help text.
func (c *Conn) Help() (io.Reader, os.Error) {
	if _, _, err := c.cmd(100, "HELP"); err != nil {
		return nil, err
	}
	return c.body(), nil
}

// nextLastStat performs the work for NEXT, LAST, and STAT.
func (c *Conn) nextLastStat(cmd, id string) (string, string, os.Error) {
	_, line, err := c.cmd(223, maybeId(cmd, id))
	if err != nil {
		return "", "", err
	}
	ss := strings.Split(line, " ", 3) // optional comment ignored
	if len(ss) < 2 {
		return "", "", ProtocolError("Bad response to " + cmd + ": " + line)
	}
	return ss[0], ss[1], nil
}

// Stat looks up the message with the given id and returns its
// message number in the current group, and vice versa.
// The returned message number can be "0" if the current group
// isn't one of the groups the message was posted to.
func (c *Conn) Stat(id string) (number, msgid string, err os.Error) {
	return c.nextLastStat("STAT", id)
}

// Last selects the previous article, returning its message number and id.
func (c *Conn) Last() (number, msgid string, err os.Error) {
	return c.nextLastStat("LAST", "")
}

// Next selects the next article, returning its message number and id.
func (c *Conn) Next() (number, msgid string, err os.Error) {
	return c.nextLastStat("NEXT", "")
}

// ArticleText returns the article named by id as an io.Reader.
// The article is in plain text format, not NNTP wire format.
func (c *Conn) ArticleText(id string) (io.Reader, os.Error) {
	if _, _, err := c.cmd(220, maybeId("ARTICLE", id)); err != nil {
		return nil, err
	}
	return c.body(), nil
}

// Article returns the article named by id as an *Article.
func (c *Conn) Article(id string) (*Article, os.Error) {
	if _, _, err := c.cmd(220, maybeId("ARTICLE", id)); err != nil {
		return nil, err
	}
	r := bufio.NewReader(c.body())
	res, err := c.readHeader(r)
	if err != nil {
		return nil, err
	}
	res.Body = r
	return res, nil
}

// HeadText returns the header for the article named by id as an io.Reader.
// The article is in plain text format, not NNTP wire format.
func (c *Conn) HeadText(id string) (io.Reader, os.Error) {
	if _, _, err := c.cmd(221, maybeId("HEAD", id)); err != nil {
		return nil, err
	}
	return c.body(), nil
}

// Head returns the header for the article named by id as an *Article.
// The Body field in the Article is nil.
func (c *Conn) Head(id string) (*Article, os.Error) {
	if _, _, err := c.cmd(221, maybeId("HEAD", id)); err != nil {
		return nil, err
	}
	return c.readHeader(bufio.NewReader(c.body()))
}

// Body returns the body for the article named by id as an io.Reader.
func (c *Conn) Body(id string) (io.Reader, os.Error) {
	if _, _, err := c.cmd(222, maybeId("BODY", id)); err != nil {
		return nil, err
	}
	return c.body(), nil
}

// RawPost reads a text-formatted article from r and posts it to the server.
func (c *Conn) RawPost(r io.Reader) os.Error {
	if _, _, err := c.cmd(3, "POST"); err != nil {
		return err
	}
	br := bufio.NewReader(r)
	eof := false
	for {
		line, err := br.ReadString('\n')
		if err == os.EOF {
			eof = true
		} else if err != nil {
			return err
		}
		if eof && len(line) == 0 {
			break
		}
		if strings.HasSuffix(line, "\n") {
			line = line[0 : len(line)-1]
		}
		var prefix string
		if strings.HasPrefix(line, ".") {
			prefix = "."
		}
		_, err = fmt.Fprintf(c.conn, "%s%s\r\n", prefix, line)
		if err != nil {
			return err
		}
		if eof {
			break
		}
	}

	if _, _, err := c.cmd(240, "."); err != nil {
		return err
	}
	return nil
}

// Post posts an article to the server.
func (c *Conn) Post(a *Article) os.Error {
	return c.RawPost(&articleReader{a: a})
}

// Quit sends the QUIT command and closes the connection to the server.
func (c *Conn) Quit() os.Error {
	_, _, err := c.cmd(0, "QUIT")
	c.conn.Close()
	c.close = true
	return err
}

// Functions after this point are mostly copy-pasted from http
// (though with some modifications). They should be factored out to
// a common library.

// Read a line of bytes (up to \n) from b.
// Give up if the line exceeds maxLineLength.
// The returned bytes are a pointer into storage in
// the bufio, so they are only valid until the next bufio read.
func readLineBytes(b *bufio.Reader) (p []byte, err os.Error) {
	if p, err = b.ReadSlice('\n'); err != nil {
		// We always know when EOF is coming.
		// If the caller asked for a line, there should be a line.
		if err == os.EOF {
			err = io.ErrUnexpectedEOF
		}
		return nil, err
	}

	// Chop off trailing white space.
	var i int
	for i = len(p); i > 0; i-- {
		if c := p[i-1]; c != ' ' && c != '\r' && c != '\t' && c != '\n' {
			break
		}
	}
	return p[0:i], nil
}

var colon = []byte{':'}

// Read a key/value pair from b.
// A key/value has the form Key: Value\r\n
// and the Value can continue on multiple lines if each continuation line
// starts with a space/tab.
func readKeyValue(b *bufio.Reader) (key, value string, err os.Error) {
	line, e := readLineBytes(b)
	if e == io.ErrUnexpectedEOF {
		return "", "", nil
	} else if e != nil {
		return "", "", e
	}
	if len(line) == 0 {
		return "", "", nil
	}

	// Scan first line for colon.
	i := bytes.Index(line, colon)
	if i < 0 {
		goto Malformed
	}

	key = string(line[0:i])
	if strings.Index(key, " ") >= 0 {
		// Key field has space - no good.
		goto Malformed
	}

	// Skip initial space before value.
	for i++; i < len(line); i++ {
		if line[i] != ' ' && line[i] != '\t' {
			break
		}
	}
	value = string(line[i:])

	// Look for extension lines, which must begin with space.
	for {
		c, e := b.ReadByte()
		if c != ' ' && c != '\t' {
			if e != os.EOF {
				b.UnreadByte()
			}
			break
		}

		// Eat leading space.
		for c == ' ' || c == '\t' {
			if c, e = b.ReadByte(); e != nil {
				if e == os.EOF {
					e = io.ErrUnexpectedEOF
				}
				return "", "", e
			}
		}
		b.UnreadByte()

		// Read the rest of the line and add to value.
		if line, e = readLineBytes(b); e != nil {
			return "", "", e
		}
		value += " " + string(line)
	}
	return key, value, nil

Malformed:
	return "", "", ProtocolError("malformed header line: " + string(line))
}

// Internal. Parses headers in NNTP articles. Most of this is stolen from the http package,
// and it should probably be split out into a generic RFC822 header-parsing package.
func (c *Conn) readHeader(r *bufio.Reader) (res *Article, err os.Error) {
	res = new(Article)
	res.Header = make(map[string][]string)
	for {
		var key, value string
		if key, value, err = readKeyValue(r); err != nil {
			return nil, err
		}
		if key == "" {
			break
		}
		key = http.CanonicalHeaderKey(key)
		// RFC 3977 says nothing about duplicate keys' values being equivalent to
		// a single key joined with commas, so we keep all values seperate.
		oldvalue, present := res.Header[key]
		if present {
			sv := vector.StringVector(oldvalue)
			sv.Push(value)
			res.Header[key] = []string(sv)
		} else {
			res.Header[key] = []string{value}
		}
	}
	return res, nil
}
