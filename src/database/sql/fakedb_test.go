// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sql

import (
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"log"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

var _ = log.Printf

// fakeDriver is a fake database that implements Go's driver.Driver
// interface, just for testing.
//
// It speaks a query language that's semantically similar to but
// syntactically different and simpler than SQL.  The syntax is as
// follows:
//
//   WIPE
//   CREATE|<tablename>|<col>=<type>,<col>=<type>,...
//     where types are: "string", [u]int{8,16,32,64}, "bool"
//   INSERT|<tablename>|col=val,col2=val2,col3=?
//   SELECT|<tablename>|projectcol1,projectcol2|filtercol=?,filtercol2=?
//
// Any of these can be preceded by PANIC|<method>|, to cause the
// named method on fakeStmt to panic.
//
// When opening a fakeDriver's database, it starts empty with no
// tables. All tables and data are stored in memory only.
type fakeDriver struct {
	mu         sync.Mutex // guards 3 following fields
	openCount  int        // conn opens
	closeCount int        // conn closes
	waitCh     chan struct{}
	waitingCh  chan struct{}
	dbs        map[string]*fakeDB
}

type fakeDB struct {
	name string

	mu      sync.Mutex
	tables  map[string]*table
	badConn bool
}

type table struct {
	mu      sync.Mutex
	colname []string
	coltype []string
	rows    []*row
}

func (t *table) columnIndex(name string) int {
	for n, nname := range t.colname {
		if name == nname {
			return n
		}
	}
	return -1
}

type row struct {
	cols []interface{} // must be same size as its table colname + coltype
}

type fakeConn struct {
	db *fakeDB // where to return ourselves to

	currTx *fakeTx

	// Stats for tests:
	mu          sync.Mutex
	stmtsMade   int
	stmtsClosed int
	numPrepare  int

	// bad connection tests; see isBad()
	bad       bool
	stickyBad bool
}

func (c *fakeConn) incrStat(v *int) {
	c.mu.Lock()
	*v++
	c.mu.Unlock()
}

type fakeTx struct {
	c *fakeConn
}

type fakeStmt struct {
	c *fakeConn
	q string // just for debugging

	cmd   string
	table string
	panic string

	closed bool

	colName      []string      // used by CREATE, INSERT, SELECT (selected columns)
	colType      []string      // used by CREATE
	colValue     []interface{} // used by INSERT (mix of strings and "?" for bound params)
	placeholders int           // used by INSERT/SELECT: number of ? params

	whereCol []string // used by SELECT (all placeholders)

	placeholderConverter []driver.ValueConverter // used by INSERT
}

var fdriver driver.Driver = &fakeDriver{}

func init() {
	Register("test", fdriver)
}

func contains(list []string, y string) bool {
	for _, x := range list {
		if x == y {
			return true
		}
	}
	return false
}

type Dummy struct {
	driver.Driver
}

func TestDrivers(t *testing.T) {
	unregisterAllDrivers()
	Register("test", fdriver)
	Register("invalid", Dummy{})
	all := Drivers()
	if len(all) < 2 || !sort.StringsAreSorted(all) || !contains(all, "test") || !contains(all, "invalid") {
		t.Fatalf("Drivers = %v, want sorted list with at least [invalid, test]", all)
	}
}

// hook to simulate connection failures
var hookOpenErr struct {
	sync.Mutex
	fn func() error
}

func setHookOpenErr(fn func() error) {
	hookOpenErr.Lock()
	defer hookOpenErr.Unlock()
	hookOpenErr.fn = fn
}

// Supports dsn forms:
//    <dbname>
//    <dbname>;<opts>  (only currently supported option is `badConn`,
//                      which causes driver.ErrBadConn to be returned on
//                      every other conn.Begin())
func (d *fakeDriver) Open(dsn string) (driver.Conn, error) {
	hookOpenErr.Lock()
	fn := hookOpenErr.fn
	hookOpenErr.Unlock()
	if fn != nil {
		if err := fn(); err != nil {
			return nil, err
		}
	}
	parts := strings.Split(dsn, ";")
	if len(parts) < 1 {
		return nil, errors.New("fakedb: no database name")
	}
	name := parts[0]

	db := d.getDB(name)

	d.mu.Lock()
	d.openCount++
	d.mu.Unlock()
	conn := &fakeConn{db: db}

	if len(parts) >= 2 && parts[1] == "badConn" {
		conn.bad = true
	}
	if d.waitCh != nil {
		d.waitingCh <- struct{}{}
		<-d.waitCh
		d.waitCh = nil
		d.waitingCh = nil
	}
	return conn, nil
}

func (d *fakeDriver) getDB(name string) *fakeDB {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.dbs == nil {
		d.dbs = make(map[string]*fakeDB)
	}
	db, ok := d.dbs[name]
	if !ok {
		db = &fakeDB{name: name}
		d.dbs[name] = db
	}
	return db
}

func (db *fakeDB) wipe() {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.tables = nil
}

func (db *fakeDB) createTable(name string, columnNames, columnTypes []string) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	if db.tables == nil {
		db.tables = make(map[string]*table)
	}
	if _, exist := db.tables[name]; exist {
		return fmt.Errorf("table %q already exists", name)
	}
	if len(columnNames) != len(columnTypes) {
		return fmt.Errorf("create table of %q len(names) != len(types): %d vs %d",
			name, len(columnNames), len(columnTypes))
	}
	db.tables[name] = &table{colname: columnNames, coltype: columnTypes}
	return nil
}

// must be called with db.mu lock held
func (db *fakeDB) table(table string) (*table, bool) {
	if db.tables == nil {
		return nil, false
	}
	t, ok := db.tables[table]
	return t, ok
}

func (db *fakeDB) columnType(table, column string) (typ string, ok bool) {
	db.mu.Lock()
	defer db.mu.Unlock()
	t, ok := db.table(table)
	if !ok {
		return
	}
	for n, cname := range t.colname {
		if cname == column {
			return t.coltype[n], true
		}
	}
	return "", false
}

func (c *fakeConn) isBad() bool {
	if c.stickyBad {
		return true
	} else if c.bad {
		// alternate between bad conn and not bad conn
		c.db.badConn = !c.db.badConn
		return c.db.badConn
	} else {
		return false
	}
}

func (c *fakeConn) Begin() (driver.Tx, error) {
	if c.isBad() {
		return nil, driver.ErrBadConn
	}
	if c.currTx != nil {
		return nil, errors.New("already in a transaction")
	}
	c.currTx = &fakeTx{c: c}
	return c.currTx, nil
}

var hookPostCloseConn struct {
	sync.Mutex
	fn func(*fakeConn, error)
}

func setHookpostCloseConn(fn func(*fakeConn, error)) {
	hookPostCloseConn.Lock()
	defer hookPostCloseConn.Unlock()
	hookPostCloseConn.fn = fn
}

var testStrictClose *testing.T

// setStrictFakeConnClose sets the t to Errorf on when fakeConn.Close
// fails to close. If nil, the check is disabled.
func setStrictFakeConnClose(t *testing.T) {
	testStrictClose = t
}

func (c *fakeConn) Close() (err error) {
	drv := fdriver.(*fakeDriver)
	defer func() {
		if err != nil && testStrictClose != nil {
			testStrictClose.Errorf("failed to close a test fakeConn: %v", err)
		}
		hookPostCloseConn.Lock()
		fn := hookPostCloseConn.fn
		hookPostCloseConn.Unlock()
		if fn != nil {
			fn(c, err)
		}
		if err == nil {
			drv.mu.Lock()
			drv.closeCount++
			drv.mu.Unlock()
		}
	}()
	if c.currTx != nil {
		return errors.New("can't close fakeConn; in a Transaction")
	}
	if c.db == nil {
		return errors.New("can't close fakeConn; already closed")
	}
	if c.stmtsMade > c.stmtsClosed {
		return errors.New("can't close; dangling statement(s)")
	}
	c.db = nil
	return nil
}

func checkSubsetTypes(args []driver.Value) error {
	for n, arg := range args {
		switch arg.(type) {
		case int64, float64, bool, nil, []byte, string, time.Time:
		default:
			return fmt.Errorf("fakedb_test: invalid argument #%d: %v, type %T", n+1, arg, arg)
		}
	}
	return nil
}

func (c *fakeConn) Exec(query string, args []driver.Value) (driver.Result, error) {
	// This is an optional interface, but it's implemented here
	// just to check that all the args are of the proper types.
	// ErrSkip is returned so the caller acts as if we didn't
	// implement this at all.
	err := checkSubsetTypes(args)
	if err != nil {
		return nil, err
	}
	return nil, driver.ErrSkip
}

func (c *fakeConn) Query(query string, args []driver.Value) (driver.Rows, error) {
	// This is an optional interface, but it's implemented here
	// just to check that all the args are of the proper types.
	// ErrSkip is returned so the caller acts as if we didn't
	// implement this at all.
	err := checkSubsetTypes(args)
	if err != nil {
		return nil, err
	}
	return nil, driver.ErrSkip
}

func errf(msg string, args ...interface{}) error {
	return errors.New("fakedb: " + fmt.Sprintf(msg, args...))
}

// parts are table|selectCol1,selectCol2|whereCol=?,whereCol2=?
// (note that where columns must always contain ? marks,
//  just a limitation for fakedb)
func (c *fakeConn) prepareSelect(stmt *fakeStmt, parts []string) (driver.Stmt, error) {
	if len(parts) != 3 {
		stmt.Close()
		return nil, errf("invalid SELECT syntax with %d parts; want 3", len(parts))
	}
	stmt.table = parts[0]
	stmt.colName = strings.Split(parts[1], ",")
	for n, colspec := range strings.Split(parts[2], ",") {
		if colspec == "" {
			continue
		}
		nameVal := strings.Split(colspec, "=")
		if len(nameVal) != 2 {
			stmt.Close()
			return nil, errf("SELECT on table %q has invalid column spec of %q (index %d)", stmt.table, colspec, n)
		}
		column, value := nameVal[0], nameVal[1]
		_, ok := c.db.columnType(stmt.table, column)
		if !ok {
			stmt.Close()
			return nil, errf("SELECT on table %q references non-existent column %q", stmt.table, column)
		}
		if value != "?" {
			stmt.Close()
			return nil, errf("SELECT on table %q has pre-bound value for where column %q; need a question mark",
				stmt.table, column)
		}
		stmt.whereCol = append(stmt.whereCol, column)
		stmt.placeholders++
	}
	return stmt, nil
}

// parts are table|col=type,col2=type2
func (c *fakeConn) prepareCreate(stmt *fakeStmt, parts []string) (driver.Stmt, error) {
	if len(parts) != 2 {
		stmt.Close()
		return nil, errf("invalid CREATE syntax with %d parts; want 2", len(parts))
	}
	stmt.table = parts[0]
	for n, colspec := range strings.Split(parts[1], ",") {
		nameType := strings.Split(colspec, "=")
		if len(nameType) != 2 {
			stmt.Close()
			return nil, errf("CREATE table %q has invalid column spec of %q (index %d)", stmt.table, colspec, n)
		}
		stmt.colName = append(stmt.colName, nameType[0])
		stmt.colType = append(stmt.colType, nameType[1])
	}
	return stmt, nil
}

// parts are table|col=?,col2=val
func (c *fakeConn) prepareInsert(stmt *fakeStmt, parts []string) (driver.Stmt, error) {
	if len(parts) != 2 {
		stmt.Close()
		return nil, errf("invalid INSERT syntax with %d parts; want 2", len(parts))
	}
	stmt.table = parts[0]
	for n, colspec := range strings.Split(parts[1], ",") {
		nameVal := strings.Split(colspec, "=")
		if len(nameVal) != 2 {
			stmt.Close()
			return nil, errf("INSERT table %q has invalid column spec of %q (index %d)", stmt.table, colspec, n)
		}
		column, value := nameVal[0], nameVal[1]
		ctype, ok := c.db.columnType(stmt.table, column)
		if !ok {
			stmt.Close()
			return nil, errf("INSERT table %q references non-existent column %q", stmt.table, column)
		}
		stmt.colName = append(stmt.colName, column)

		if value != "?" {
			var subsetVal interface{}
			// Convert to driver subset type
			switch ctype {
			case "string":
				subsetVal = []byte(value)
			case "blob":
				subsetVal = []byte(value)
			case "int32":
				i, err := strconv.Atoi(value)
				if err != nil {
					stmt.Close()
					return nil, errf("invalid conversion to int32 from %q", value)
				}
				subsetVal = int64(i) // int64 is a subset type, but not int32
			default:
				stmt.Close()
				return nil, errf("unsupported conversion for pre-bound parameter %q to type %q", value, ctype)
			}
			stmt.colValue = append(stmt.colValue, subsetVal)
		} else {
			stmt.placeholders++
			stmt.placeholderConverter = append(stmt.placeholderConverter, converterForType(ctype))
			stmt.colValue = append(stmt.colValue, "?")
		}
	}
	return stmt, nil
}

// hook to simulate broken connections
var hookPrepareBadConn func() bool

func (c *fakeConn) Prepare(query string) (driver.Stmt, error) {
	c.numPrepare++
	if c.db == nil {
		panic("nil c.db; conn = " + fmt.Sprintf("%#v", c))
	}

	if c.stickyBad || (hookPrepareBadConn != nil && hookPrepareBadConn()) {
		return nil, driver.ErrBadConn
	}

	parts := strings.Split(query, "|")
	if len(parts) < 1 {
		return nil, errf("empty query")
	}
	stmt := &fakeStmt{q: query, c: c}
	if len(parts) >= 3 && parts[0] == "PANIC" {
		stmt.panic = parts[1]
		parts = parts[2:]
	}
	cmd := parts[0]
	stmt.cmd = cmd
	parts = parts[1:]

	c.incrStat(&c.stmtsMade)
	switch cmd {
	case "WIPE":
		// Nothing
	case "SELECT":
		return c.prepareSelect(stmt, parts)
	case "CREATE":
		return c.prepareCreate(stmt, parts)
	case "INSERT":
		return c.prepareInsert(stmt, parts)
	case "NOSERT":
		// Do all the prep-work like for an INSERT but don't actually insert the row.
		// Used for some of the concurrent tests.
		return c.prepareInsert(stmt, parts)
	default:
		stmt.Close()
		return nil, errf("unsupported command type %q", cmd)
	}
	return stmt, nil
}

func (s *fakeStmt) ColumnConverter(idx int) driver.ValueConverter {
	if s.panic == "ColumnConverter" {
		panic(s.panic)
	}
	if len(s.placeholderConverter) == 0 {
		return driver.DefaultParameterConverter
	}
	return s.placeholderConverter[idx]
}

func (s *fakeStmt) Close() error {
	if s.panic == "Close" {
		panic(s.panic)
	}
	if s.c == nil {
		panic("nil conn in fakeStmt.Close")
	}
	if s.c.db == nil {
		panic("in fakeStmt.Close, conn's db is nil (already closed)")
	}
	if !s.closed {
		s.c.incrStat(&s.c.stmtsClosed)
		s.closed = true
	}
	return nil
}

var errClosed = errors.New("fakedb: statement has been closed")

// hook to simulate broken connections
var hookExecBadConn func() bool

func (s *fakeStmt) Exec(args []driver.Value) (driver.Result, error) {
	if s.panic == "Exec" {
		panic(s.panic)
	}
	if s.closed {
		return nil, errClosed
	}

	if s.c.stickyBad || (hookExecBadConn != nil && hookExecBadConn()) {
		return nil, driver.ErrBadConn
	}

	err := checkSubsetTypes(args)
	if err != nil {
		return nil, err
	}

	db := s.c.db
	switch s.cmd {
	case "WIPE":
		db.wipe()
		return driver.ResultNoRows, nil
	case "CREATE":
		if err := db.createTable(s.table, s.colName, s.colType); err != nil {
			return nil, err
		}
		return driver.ResultNoRows, nil
	case "INSERT":
		return s.execInsert(args, true)
	case "NOSERT":
		// Do all the prep-work like for an INSERT but don't actually insert the row.
		// Used for some of the concurrent tests.
		return s.execInsert(args, false)
	}
	fmt.Printf("EXEC statement, cmd=%q: %#v\n", s.cmd, s)
	return nil, fmt.Errorf("unimplemented statement Exec command type of %q", s.cmd)
}

// When doInsert is true, add the row to the table.
// When doInsert is false do prep-work and error checking, but don't
// actually add the row to the table.
func (s *fakeStmt) execInsert(args []driver.Value, doInsert bool) (driver.Result, error) {
	db := s.c.db
	if len(args) != s.placeholders {
		panic("error in pkg db; should only get here if size is correct")
	}
	db.mu.Lock()
	t, ok := db.table(s.table)
	db.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("fakedb: table %q doesn't exist", s.table)
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	var cols []interface{}
	if doInsert {
		cols = make([]interface{}, len(t.colname))
	}
	argPos := 0
	for n, colname := range s.colName {
		colidx := t.columnIndex(colname)
		if colidx == -1 {
			return nil, fmt.Errorf("fakedb: column %q doesn't exist or dropped since prepared statement was created", colname)
		}
		var val interface{}
		if strvalue, ok := s.colValue[n].(string); ok && strvalue == "?" {
			val = args[argPos]
			argPos++
		} else {
			val = s.colValue[n]
		}
		if doInsert {
			cols[colidx] = val
		}
	}

	if doInsert {
		t.rows = append(t.rows, &row{cols: cols})
	}
	return driver.RowsAffected(1), nil
}

// hook to simulate broken connections
var hookQueryBadConn func() bool

func (s *fakeStmt) Query(args []driver.Value) (driver.Rows, error) {
	if s.panic == "Query" {
		panic(s.panic)
	}
	if s.closed {
		return nil, errClosed
	}

	if s.c.stickyBad || (hookQueryBadConn != nil && hookQueryBadConn()) {
		return nil, driver.ErrBadConn
	}

	err := checkSubsetTypes(args)
	if err != nil {
		return nil, err
	}

	db := s.c.db
	if len(args) != s.placeholders {
		panic("error in pkg db; should only get here if size is correct")
	}

	db.mu.Lock()
	t, ok := db.table(s.table)
	db.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("fakedb: table %q doesn't exist", s.table)
	}

	if s.table == "magicquery" {
		if len(s.whereCol) == 2 && s.whereCol[0] == "op" && s.whereCol[1] == "millis" {
			if args[0] == "sleep" {
				time.Sleep(time.Duration(args[1].(int64)) * time.Millisecond)
			}
		}
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	colIdx := make(map[string]int) // select column name -> column index in table
	for _, name := range s.colName {
		idx := t.columnIndex(name)
		if idx == -1 {
			return nil, fmt.Errorf("fakedb: unknown column name %q", name)
		}
		colIdx[name] = idx
	}

	mrows := []*row{}
rows:
	for _, trow := range t.rows {
		// Process the where clause, skipping non-match rows. This is lazy
		// and just uses fmt.Sprintf("%v") to test equality. Good enough
		// for test code.
		for widx, wcol := range s.whereCol {
			idx := t.columnIndex(wcol)
			if idx == -1 {
				return nil, fmt.Errorf("db: invalid where clause column %q", wcol)
			}
			tcol := trow.cols[idx]
			if bs, ok := tcol.([]byte); ok {
				// lazy hack to avoid sprintf %v on a []byte
				tcol = string(bs)
			}
			if fmt.Sprintf("%v", tcol) != fmt.Sprintf("%v", args[widx]) {
				continue rows
			}
		}
		mrow := &row{cols: make([]interface{}, len(s.colName))}
		for seli, name := range s.colName {
			mrow.cols[seli] = trow.cols[colIdx[name]]
		}
		mrows = append(mrows, mrow)
	}

	cursor := &rowsCursor{
		pos:    -1,
		rows:   mrows,
		cols:   s.colName,
		errPos: -1,
	}
	return cursor, nil
}

func (s *fakeStmt) NumInput() int {
	if s.panic == "NumInput" {
		panic(s.panic)
	}
	return s.placeholders
}

// hook to simulate broken connections
var hookCommitBadConn func() bool

func (tx *fakeTx) Commit() error {
	tx.c.currTx = nil
	if hookCommitBadConn != nil && hookCommitBadConn() {
		return driver.ErrBadConn
	}
	return nil
}

// hook to simulate broken connections
var hookRollbackBadConn func() bool

func (tx *fakeTx) Rollback() error {
	tx.c.currTx = nil
	if hookRollbackBadConn != nil && hookRollbackBadConn() {
		return driver.ErrBadConn
	}
	return nil
}

type rowsCursor struct {
	cols   []string
	pos    int
	rows   []*row
	closed bool

	// errPos and err are for making Next return early with error.
	errPos int
	err    error

	// a clone of slices to give out to clients, indexed by the
	// the original slice's first byte address.  we clone them
	// just so we're able to corrupt them on close.
	bytesClone map[*byte][]byte
}

func (rc *rowsCursor) Close() error {
	if !rc.closed {
		for _, bs := range rc.bytesClone {
			bs[0] = 255 // first byte corrupted
		}
	}
	rc.closed = true
	return nil
}

func (rc *rowsCursor) Columns() []string {
	return rc.cols
}

var rowsCursorNextHook func(dest []driver.Value) error

func (rc *rowsCursor) Next(dest []driver.Value) error {
	if rowsCursorNextHook != nil {
		return rowsCursorNextHook(dest)
	}

	if rc.closed {
		return errors.New("fakedb: cursor is closed")
	}
	rc.pos++
	if rc.pos == rc.errPos {
		return rc.err
	}
	if rc.pos >= len(rc.rows) {
		return io.EOF // per interface spec
	}
	for i, v := range rc.rows[rc.pos].cols {
		// TODO(bradfitz): convert to subset types? naah, I
		// think the subset types should only be input to
		// driver, but the sql package should be able to handle
		// a wider range of types coming out of drivers. all
		// for ease of drivers, and to prevent drivers from
		// messing up conversions or doing them differently.
		dest[i] = v

		if bs, ok := v.([]byte); ok {
			if rc.bytesClone == nil {
				rc.bytesClone = make(map[*byte][]byte)
			}
			clone, ok := rc.bytesClone[&bs[0]]
			if !ok {
				clone = make([]byte, len(bs))
				copy(clone, bs)
				rc.bytesClone[&bs[0]] = clone
			}
			dest[i] = clone
		}
	}
	return nil
}

// fakeDriverString is like driver.String, but indirects pointers like
// DefaultValueConverter.
//
// This could be surprising behavior to retroactively apply to
// driver.String now that Go1 is out, but this is convenient for
// our TestPointerParamsAndScans.
//
type fakeDriverString struct{}

func (fakeDriverString) ConvertValue(v interface{}) (driver.Value, error) {
	switch c := v.(type) {
	case string, []byte:
		return v, nil
	case *string:
		if c == nil {
			return nil, nil
		}
		return *c, nil
	}
	return fmt.Sprintf("%v", v), nil
}

func converterForType(typ string) driver.ValueConverter {
	switch typ {
	case "bool":
		return driver.Bool
	case "nullbool":
		return driver.Null{Converter: driver.Bool}
	case "int32":
		return driver.Int32
	case "string":
		return driver.NotNull{Converter: fakeDriverString{}}
	case "nullstring":
		return driver.Null{Converter: fakeDriverString{}}
	case "int64":
		// TODO(coopernurse): add type-specific converter
		return driver.NotNull{Converter: driver.DefaultParameterConverter}
	case "nullint64":
		// TODO(coopernurse): add type-specific converter
		return driver.Null{Converter: driver.DefaultParameterConverter}
	case "float64":
		// TODO(coopernurse): add type-specific converter
		return driver.NotNull{Converter: driver.DefaultParameterConverter}
	case "nullfloat64":
		// TODO(coopernurse): add type-specific converter
		return driver.Null{Converter: driver.DefaultParameterConverter}
	case "datetime":
		return driver.DefaultParameterConverter
	}
	panic("invalid fakedb column type of " + typ)
}
