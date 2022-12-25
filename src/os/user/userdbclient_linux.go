// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package user

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"unicode/utf16"
	"unicode/utf8"
)

const (
	// Well known multiplexer service.
	svcMultiplexer = "io.systemd.Multiplexer"

	userdbNamespace = "io.systemd.UserDatabase"

	// io.systemd.UserDatabase VARLINK interface methods.
	mGetGroupRecord = userdbNamespace + ".GetGroupRecord"
	mGetUserRecord  = userdbNamespace + ".GetUserRecord"
	mGetMemberships = userdbNamespace + ".GetMemberships"

	// io.systemd.UserDatabase VARLINK interface errors.
	errNoRecordFound       = userdbNamespace + ".NoRecordFound"
	errServiceNotAvailable = userdbNamespace + ".ServiceNotAvailable"
)

func init() {
	defaultUserdbClient.dir = "/run/systemd/userdb"
}

// userdbCall represents a VARLINK service call sent to systemd-userdb.
// method is the VARLINK method to call.
// parameters are the VARLINK parameters to pass.
// more indicates if more responses are expected.
// fastest indicates if only the fastest response should be returned.
type userdbCall struct {
	method     string
	parameters callParameters
	more       bool
	fastest    bool
}

func (u userdbCall) marshalJSON(service string) ([]byte, error) {
	params, err := u.parameters.marshalJSON(service)
	if err != nil {
		return nil, err
	}
	var data bytes.Buffer
	data.WriteString(`{"method":"`)
	data.WriteString(u.method)
	data.WriteString(`","parameters":`)
	data.Write(params)
	if u.more {
		data.WriteString(`,"more":true`)
	}
	data.WriteString(`}`)
	return data.Bytes(), nil
}

type callParameters struct {
	uid       *int64
	userName  string
	gid       *int64
	groupName string
}

func (c callParameters) marshalJSON(service string) ([]byte, error) {
	var data bytes.Buffer
	data.WriteString(`{"service":"`)
	data.WriteString(service)
	data.WriteString(`"`)
	if c.uid != nil {
		data.WriteString(`,"uid":`)
		data.WriteString(strconv.FormatInt(*c.uid, 10))
	}
	if c.userName != "" {
		data.WriteString(`,"userName":"`)
		data.WriteString(c.userName)
		data.WriteString(`"`)
	}
	if c.gid != nil {
		data.WriteString(`,"gid":`)
		data.WriteString(strconv.FormatInt(*c.gid, 10))
	}
	if c.groupName != "" {
		data.WriteString(`,"groupName":"`)
		data.WriteString(c.groupName)
		data.WriteString(`"`)
	}
	data.WriteString(`}`)
	return data.Bytes(), nil
}

type userdbReply struct {
	continues bool
	errorStr  string
}

func (u *userdbReply) unmarshalJSON(data []byte) error {
	var (
		kContinues = []byte(`"continues"`)
		kError     = []byte(`"error"`)
	)
	if i := bytes.Index(data, kContinues); i != -1 {
		continues, err := parseJSONBoolean(data[i+len(kContinues):])
		if err != nil {
			return err
		}
		u.continues = continues
	}
	if i := bytes.Index(data, kError); i != -1 {
		errStr, err := parseJSONString(data[i+len(kError):])
		if err != nil {
			return err
		}
		u.errorStr = errStr
	}
	return nil
}

// response is the parsed reply from a method call to systemd-userdb.
// data is one or more VARLINK response parameters separated by 0.
// handled indicates if the call was handled by systemd-userdb.
// err is any error encountered.
type response struct {
	data    []byte
	handled bool
	err     error
}

// querySocket calls the io.systemd.UserDatabase VARLINK interface at sock with request.
// Multiple replies can be read by setting more to true in the request.
// Reply parameters are accumulated separated by 0, if there are many.
// Replies with io.systemd.UserDatabase.NoRecordFound errors are skipped.
// Other UserDatabase errors are returned as is.
// If the socket does not exist, or if the io.systemd.UserDatabase.ServiceNotAvailable
// error is seen in a response, the query is considered unhandled.
func querySocket(ctx context.Context, sock string, request []byte) response {
	sockFd, err := syscall.Socket(syscall.AF_UNIX, syscall.SOCK_STREAM, 0)
	if err != nil {
		return response{err: err}
	}
	defer syscall.Close(sockFd)
	if err := syscall.Connect(sockFd, &syscall.SockaddrUnix{Name: sock}); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return response{err: err}
		}
		return response{handled: true, err: err}
	}

	// Null terminate request.
	if request[len(request)-1] != 0 {
		request = append(request, 0)
	}

	// Write request to socket.
	written := 0
	for written < len(request) {
		if ctx.Err() != nil {
			return response{handled: true, err: ctx.Err()}
		}
		if n, err := syscall.Write(sockFd, request[written:]); err != nil {
			return response{handled: true, err: err}
		} else {
			written += n
		}
	}

	// Read response.
	var resp bytes.Buffer
	for {
		if ctx.Err() != nil {
			return response{handled: true, err: ctx.Err()}
		}
		buf := make([]byte, 4096)
		if n, err := syscall.Read(sockFd, buf); err != nil {
			return response{handled: true, err: err}
		} else if n > 0 {
			resp.Write(buf[:n])
			if buf[n-1] == 0 {
				break
			}
		} else {
			// EOF
			break
		}
	}

	if resp.Len() == 0 {
		return response{handled: true}
	}

	buf := resp.Bytes()
	// Remove trailing 0.
	buf = buf[:len(buf)-1]
	// Split into VARLINK messages.
	msgs := bytes.Split(buf, []byte{0})

	// Parse VARLINK messages.
	for _, m := range msgs {
		var resp userdbReply
		if err := resp.unmarshalJSON(m); err != nil {
			return response{handled: true, err: err}
		}
		// Handle VARLINK message errors.
		switch e := resp.errorStr; e {
		case "":
		case errNoRecordFound: // Ignore not found error.
			continue
		case errServiceNotAvailable:
			return response{}
		default:
			return response{handled: true, err: errors.New(e)}
		}
		if !resp.continues {
			break
		}
	}
	return response{data: buf, handled: true, err: ctx.Err()}
}

// queryMany calls the io.systemd.UserDatabase VARLINK interface on many services at once.
// ss is a slice of userdb services to call. Each service must have a socket in cl.dir.
// c is sent to all services in ss. If c.fastest is true, only the fastest reply is read.
// Otherwise all replies are aggregated. um is called with aggregated reply parameters.
// queryMany returns the first error encountered. The first result is false if no userdb
// socket is available or if all requests time out.
func (cl userdbClient) queryMany(ctx context.Context, ss []string, c *userdbCall, um jsonUnmarshaler) (bool, error) {
	responseCh := make(chan response, len(ss))

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	// Query all services in parallel.
	var workers sync.WaitGroup
	for _, svc := range ss {
		data, err := c.marshalJSON(svc)
		if err != nil {
			return true, err
		}
		// Spawn worker to query service.
		workers.Add(1)
		go func(sock string, data []byte) {
			defer workers.Done()
			responseCh <- querySocket(ctx, sock, data)
		}(cl.dir+"/"+svc, data)
	}

	go func() {
		// Clean up workers.
		workers.Wait()
		close(responseCh)
	}()

	var result bytes.Buffer
	var notOk int
RecvResponses:
	for {
		select {
		case resp, ok := <-responseCh:
			if !ok {
				// Responses channel is closed so stop reading.
				break RecvResponses
			}
			if resp.err != nil {
				// querySocket only returns unrecoverable errors,
				// so return the first one received.
				return true, resp.err
			}
			if !resp.handled {
				notOk++
				continue
			}

			first := result.Len() == 0
			result.Write(resp.data)
			if first && c.fastest {
				// Return the fastest response.
				break RecvResponses
			}
		case <-ctx.Done():
			// If requests time out, userdb is unavailable.
			return ctx.Err() != context.DeadlineExceeded, nil
		}
	}
	// If all sockets are not ok, userdb is unavailable.
	if notOk == len(ss) {
		return false, nil
	}
	return true, um.unmarshalJSON(result.Bytes())
}

// services enumerates userdb service sockets in dir.
// If ok is false, io.systemd.UserDatabase service does not exist.
func (cl userdbClient) services() (s []string, ok bool, err error) {
	var entries []fs.DirEntry
	if entries, err = os.ReadDir(cl.dir); err != nil {
		ok = !os.IsNotExist(err)
		return
	}
	ok = true
	for _, ent := range entries {
		s = append(s, ent.Name())
	}
	return
}

// query looks up users/groups on the io.systemd.UserDatabase VARLINK interface.
// If the multiplexer service is available, the call is sent only to it.
// Otherwise, the call is sent simultaneously to all UserDatabase services in cl.dir.
// The fastest reply is read and parsed. All other requests are cancelled.
// If the service is unavailable, the first result is false.
// The service is considered unavailable if the requests time-out as well.
func (cl userdbClient) query(ctx context.Context, call *userdbCall, um jsonUnmarshaler) (bool, error) {
	services := []string{svcMultiplexer}
	if _, err := os.Stat(cl.dir + "/" + svcMultiplexer); err != nil {
		// No mux service so call all available services.
		var ok bool
		if services, ok, err = cl.services(); !ok || err != nil {
			return ok, err
		}
	}
	call.fastest = true
	if ok, err := cl.queryMany(ctx, services, call, um); !ok || err != nil {
		return ok, err
	}
	return true, nil
}

type jsonUnmarshaler interface {
	unmarshalJSON([]byte) error
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n'
}

// findElementStart returns a slice of r that starts at the next JSON element.
// It skips over valid JSON space characters and checks for the colon separator.
func findElementStart(r []byte) ([]byte, error) {
	var idx int
	var b byte
	colon := byte(':')
	var seenColon bool
	for idx, b = range r {
		if isSpace(b) {
			continue
		}
		if !seenColon && b == colon {
			seenColon = true
			continue
		}
		// Spotted colon and b is not a space, so value starts here.
		if seenColon {
			break
		}
		return nil, errors.New("expected colon, got invalid character: " + string(b))
	}
	if !seenColon {
		return nil, errors.New("expected colon, got end of input")
	}
	return r[idx:], nil
}

// parseJSONString reads a JSON string from r.
func parseJSONString(r []byte) (string, error) {
	r, err := findElementStart(r)
	if err != nil {
		return "", err
	}
	// Smallest valid string is `""`.
	if l := len(r); l < 2 {
		return "", errors.New("unexpected end of input")
	} else if l == 2 {
		if bytes.Equal(r, []byte(`""`)) {
			return "", nil
		}
		return "", errors.New("invalid string")
	}

	if c := r[0]; c != '"' {
		return "", errors.New(`expected " got ` + string(c))
	}
	// Advance over opening quote.
	r = r[1:]

	var value strings.Builder
	var inEsc bool
	var inUEsc bool
	var strEnds bool
	reader := bytes.NewReader(r)
	for {
		if value.Len() > 4096 {
			return "", errors.New("string too large")
		}

		// Parse unicode escape sequences.
		if inUEsc {
			maybeRune := make([]byte, 4)
			n, err := reader.Read(maybeRune)
			if err != nil || n != 4 {
				return "", fmt.Errorf("invalid unicode escape sequence \\u%s", string(maybeRune))
			}
			prn, err := strconv.ParseUint(string(maybeRune), 16, 32)
			if err != nil {
				return "", fmt.Errorf("invalid unicode escape sequence \\u%s", string(maybeRune))
			}
			rn := rune(prn)
			if !utf16.IsSurrogate(rn) {
				value.WriteRune(rn)
				inUEsc = false
				continue
			}
			// rn maybe a high surrogate; read the low surrogate.
			maybeRune = make([]byte, 6)
			n, err = reader.Read(maybeRune)
			if err != nil || n != 6 || maybeRune[0] != '\\' || maybeRune[1] != 'u' {
				// Not a valid UTF-16 surrogate pair.
				if _, err := reader.Seek(int64(-n), io.SeekCurrent); err != nil {
					return "", err
				}
				// Invalid low surrogate; write the replacement character.
				value.WriteRune(utf8.RuneError)
			} else {
				rn1, err := strconv.ParseUint(string(maybeRune[2:]), 16, 32)
				if err != nil {
					return "", fmt.Errorf("invalid unicode escape sequence %s", string(maybeRune))
				}
				// Check if rn and rn1 are valid UTF-16 surrogate pairs.
				if dec := utf16.DecodeRune(rn, rune(rn1)); dec != utf8.RuneError {
					n = utf8.EncodeRune(maybeRune, dec)
					// Write the decoded rune.
					value.Write(maybeRune[:n])
				}
			}
			inUEsc = false
			continue
		}

		if inEsc {
			b, err := reader.ReadByte()
			if err != nil {
				return "", err
			}
			switch b {
			case 'b':
				value.WriteByte('\b')
			case 'f':
				value.WriteByte('\f')
			case 'n':
				value.WriteByte('\n')
			case 'r':
				value.WriteByte('\r')
			case 't':
				value.WriteByte('\t')
			case 'u':
				inUEsc = true
			case '/':
				value.WriteByte('/')
			case '\\':
				value.WriteByte('\\')
			case '"':
				value.WriteByte('"')
			default:
				return "", errors.New("unexpected character in escape sequence " + string(b))
			}
			inEsc = false
			continue
		} else {
			rn, _, err := reader.ReadRune()
			if err != nil {
				if err == io.EOF {
					break
				}
				return "", err
			}
			if rn == '\\' {
				inEsc = true
				continue
			}
			if rn == '"' {
				// String ends on un-escaped quote.
				strEnds = true
				break
			}
			value.WriteRune(rn)
		}
	}
	if !strEnds {
		return "", errors.New("unexpected end of input")
	}
	return value.String(), nil
}

// parseJSONInt64 reads a 64 bit integer from r.
func parseJSONInt64(r []byte) (int64, error) {
	r, err := findElementStart(r)
	if err != nil {
		return 0, err
	}
	var num strings.Builder
	for _, b := range r {
		// int64 max is 19 digits long.
		if num.Len() == 20 {
			return 0, errors.New("number too large")
		}
		if strings.ContainsRune("0123456789", rune(b)) {
			num.WriteByte(b)
		} else {
			break
		}
	}
	n, err := strconv.ParseInt(num.String(), 10, 64)
	return int64(n), err
}

// parseJSONBoolean reads a boolean from r.
func parseJSONBoolean(r []byte) (bool, error) {
	r, err := findElementStart(r)
	if err != nil {
		return false, err
	}
	if bytes.HasPrefix(r, []byte("true")) {
		return true, nil
	}
	if bytes.HasPrefix(r, []byte("false")) {
		return false, nil
	}
	return false, errors.New("unable to parse boolean value")
}

type groupRecord struct {
	groupName string
	gid       int64
}

func (g *groupRecord) unmarshalJSON(data []byte) error {
	var (
		kGroupName = []byte(`"groupName"`)
		kGid       = []byte(`"gid"`)
	)
	if i := bytes.Index(data, kGroupName); i != -1 {
		groupname, err := parseJSONString(data[i+len(kGroupName):])
		if err != nil {
			return err
		}
		g.groupName = groupname
	}
	if i := bytes.Index(data, kGid); i != -1 {
		gid, err := parseJSONInt64(data[i+len(kGid):])
		if err != nil {
			return err
		}
		g.gid = gid
	}
	return nil
}

// queryGroupDb queries the userdb interface for a gid, groupname, or both.
func (cl userdbClient) queryGroupDb(ctx context.Context, gid *int64, groupname string) (*Group, bool, error) {
	group := groupRecord{}
	request := userdbCall{
		method:     mGetGroupRecord,
		parameters: callParameters{gid: gid, groupName: groupname},
	}
	if ok, err := cl.query(ctx, &request, &group); !ok || err != nil {
		return nil, ok, fmt.Errorf("error querying systemd-userdb group record: %s", err)
	}
	return &Group{
		Name: group.groupName,
		Gid:  strconv.FormatInt(group.gid, 10),
	}, true, nil
}

type userRecord struct {
	userName      string
	realName      string
	uid           int64
	gid           int64
	homeDirectory string
}

func (u *userRecord) unmarshalJSON(data []byte) error {
	var (
		kUserName      = []byte(`"userName"`)
		kRealName      = []byte(`"realName"`)
		kUid           = []byte(`"uid"`)
		kGid           = []byte(`"gid"`)
		kHomeDirectory = []byte(`"homeDirectory"`)
	)
	if i := bytes.Index(data, kUserName); i != -1 {
		username, err := parseJSONString(data[i+len(kUserName):])
		if err != nil {
			return err
		}
		u.userName = username
	}
	if i := bytes.Index(data, kRealName); i != -1 {
		realname, err := parseJSONString(data[i+len(kRealName):])
		if err != nil {
			return err
		}
		u.realName = realname
	}
	if i := bytes.Index(data, kUid); i != -1 {
		uid, err := parseJSONInt64(data[i+len(kUid):])
		if err != nil {
			return err
		}
		u.uid = uid
	}
	if i := bytes.Index(data, kGid); i != -1 {
		gid, err := parseJSONInt64(data[i+len(kGid):])
		if err != nil {
			return err
		}
		u.gid = gid
	}
	if i := bytes.Index(data, kHomeDirectory); i != -1 {
		homedir, err := parseJSONString(data[i+len(kHomeDirectory):])
		if err != nil {
			return err
		}
		u.homeDirectory = homedir
	}
	return nil
}

// queryUserDb queries the userdb interface for a uid, username, or both.
func (cl userdbClient) queryUserDb(ctx context.Context, uid *int64, username string) (*User, bool, error) {
	user := userRecord{}
	request := userdbCall{
		method: mGetUserRecord,
		parameters: callParameters{
			uid:      uid,
			userName: username,
		},
	}
	if ok, err := cl.query(ctx, &request, &user); !ok || err != nil {
		return nil, ok, fmt.Errorf("error querying systemd-userdb user record: %s", err)
	}
	return &User{
		Uid:      strconv.FormatInt(user.uid, 10),
		Gid:      strconv.FormatInt(user.gid, 10),
		Username: user.userName,
		Name:     user.realName,
		HomeDir:  user.homeDirectory,
	}, true, nil
}

func (cl userdbClient) lookupGroup(ctx context.Context, groupname string) (*Group, bool, error) {
	return cl.queryGroupDb(ctx, nil, groupname)
}

func (cl userdbClient) lookupGroupId(ctx context.Context, id string) (*Group, bool, error) {
	gid, err := strconv.ParseInt(id, 10, 64)
	if err != nil {
		return nil, true, err
	}
	return cl.queryGroupDb(ctx, &gid, "")
}

func (cl userdbClient) lookupUser(ctx context.Context, username string) (*User, bool, error) {
	return cl.queryUserDb(ctx, nil, username)
}

func (cl userdbClient) lookupUserId(ctx context.Context, id string) (*User, bool, error) {
	uid, err := strconv.ParseInt(id, 10, 64)
	if err != nil {
		return nil, true, err
	}
	return cl.queryUserDb(ctx, &uid, "")
}

type memberships struct {
	// Keys are groupNames and values are sets of userNames.
	groupUsers map[string]map[string]struct{}
}

// unmarshalJSON expects many (userName, groupName) records separated by a null byte.
// This is used to build a membership map.
func (m *memberships) unmarshalJSON(data []byte) error {
	if m.groupUsers == nil {
		m.groupUsers = make(map[string]map[string]struct{})
	}
	var (
		kUserName  = []byte(`"userName"`)
		kGroupName = []byte(`"groupName"`)
	)
	// Split records by null terminator.
	records := bytes.Split(data, []byte{byte(0)})
	for _, rec := range records {
		if len(rec) == 0 {
			continue
		}
		var groupName string
		var userName string
		var err error
		if i := bytes.Index(rec, kGroupName); i != -1 {
			if groupName, err = parseJSONString(rec[i+len(kGroupName):]); err != nil {
				return err
			}
		}
		if i := bytes.Index(rec, kUserName); i != -1 {
			if userName, err = parseJSONString(rec[i+len(kUserName):]); err != nil {
				return err
			}
		}
		// Associate userName with groupName.
		if groupName != "" && userName != "" {
			if _, ok := m.groupUsers[groupName]; ok {
				m.groupUsers[groupName][userName] = struct{}{}
			} else {
				m.groupUsers[groupName] = map[string]struct{}{userName: {}}
			}
		}
	}
	return nil
}

func (cl userdbClient) lookupGroupIds(ctx context.Context, username string) ([]string, bool, error) {
	services, ok, err := cl.services()
	if !ok || err != nil {
		return nil, ok, err
	}
	// Fetch group memberships for username.
	var ms memberships
	request := userdbCall{
		method:     mGetMemberships,
		parameters: callParameters{userName: username},
		more:       true,
	}
	if ok, err := cl.queryMany(ctx, services, &request, &ms); !ok || err != nil {
		return nil, ok, fmt.Errorf("error querying systemd-userdb memberships record: %s", err)
	}
	// Fetch user group gid.
	var group groupRecord
	request = userdbCall{
		method:     mGetGroupRecord,
		parameters: callParameters{groupName: username},
	}
	if ok, err := cl.query(ctx, &request, &group); !ok || err != nil {
		return nil, ok, err
	}
	gids := []string{strconv.FormatInt(group.gid, 10)}

	// Fetch group records for each group.
	for g := range ms.groupUsers {
		var group groupRecord
		request.parameters.groupName = g
		// Query group for gid.
		if ok, err := cl.query(ctx, &request, &group); !ok || err != nil {
			return nil, ok, fmt.Errorf("error querying systemd-userdb group record: %s", err)
		}
		gids = append(gids, strconv.FormatInt(group.gid, 10))
	}
	return gids, true, nil
}
