// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sql provides a generic interface around SQL (or SQL-like)
// databases.
//
// The sql package must be used in conjunction with a database driver.
// See http://golang.org/s/sqldrivers for a list of drivers.
package sql

import (
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"runtime"
	"sync"
)

var drivers = make(map[string]driver.Driver)

// Register makes a database driver available by the provided name.
// If Register is called twice with the same name or if driver is nil,
// it panics.
func Register(name string, driver driver.Driver) {
	if driver == nil {
		panic("sql: Register driver is nil")
	}
	if _, dup := drivers[name]; dup {
		panic("sql: Register called twice for driver " + name)
	}
	drivers[name] = driver
}

// RawBytes is a byte slice that holds a reference to memory owned by
// the database itself. After a Scan into a RawBytes, the slice is only
// valid until the next call to Next, Scan, or Close.
type RawBytes []byte

// NullString represents a string that may be null.
// NullString implements the Scanner interface so
// it can be used as a scan destination:
//
//  var s NullString
//  err := db.QueryRow("SELECT name FROM foo WHERE id=?", id).Scan(&s)
//  ...
//  if s.Valid {
//     // use s.String
//  } else {
//     // NULL value
//  }
//
type NullString struct {
	String string
	Valid  bool // Valid is true if String is not NULL
}

// Scan implements the Scanner interface.
func (ns *NullString) Scan(value interface{}) error {
	if value == nil {
		ns.String, ns.Valid = "", false
		return nil
	}
	ns.Valid = true
	return convertAssign(&ns.String, value)
}

// Value implements the driver Valuer interface.
func (ns NullString) Value() (driver.Value, error) {
	if !ns.Valid {
		return nil, nil
	}
	return ns.String, nil
}

// NullInt64 represents an int64 that may be null.
// NullInt64 implements the Scanner interface so
// it can be used as a scan destination, similar to NullString.
type NullInt64 struct {
	Int64 int64
	Valid bool // Valid is true if Int64 is not NULL
}

// Scan implements the Scanner interface.
func (n *NullInt64) Scan(value interface{}) error {
	if value == nil {
		n.Int64, n.Valid = 0, false
		return nil
	}
	n.Valid = true
	return convertAssign(&n.Int64, value)
}

// Value implements the driver Valuer interface.
func (n NullInt64) Value() (driver.Value, error) {
	if !n.Valid {
		return nil, nil
	}
	return n.Int64, nil
}

// NullFloat64 represents a float64 that may be null.
// NullFloat64 implements the Scanner interface so
// it can be used as a scan destination, similar to NullString.
type NullFloat64 struct {
	Float64 float64
	Valid   bool // Valid is true if Float64 is not NULL
}

// Scan implements the Scanner interface.
func (n *NullFloat64) Scan(value interface{}) error {
	if value == nil {
		n.Float64, n.Valid = 0, false
		return nil
	}
	n.Valid = true
	return convertAssign(&n.Float64, value)
}

// Value implements the driver Valuer interface.
func (n NullFloat64) Value() (driver.Value, error) {
	if !n.Valid {
		return nil, nil
	}
	return n.Float64, nil
}

// NullBool represents a bool that may be null.
// NullBool implements the Scanner interface so
// it can be used as a scan destination, similar to NullString.
type NullBool struct {
	Bool  bool
	Valid bool // Valid is true if Bool is not NULL
}

// Scan implements the Scanner interface.
func (n *NullBool) Scan(value interface{}) error {
	if value == nil {
		n.Bool, n.Valid = false, false
		return nil
	}
	n.Valid = true
	return convertAssign(&n.Bool, value)
}

// Value implements the driver Valuer interface.
func (n NullBool) Value() (driver.Value, error) {
	if !n.Valid {
		return nil, nil
	}
	return n.Bool, nil
}

// Scanner is an interface used by Scan.
type Scanner interface {
	// Scan assigns a value from a database driver.
	//
	// The src value will be of one of the following restricted
	// set of types:
	//
	//    int64
	//    float64
	//    bool
	//    []byte
	//    string
	//    time.Time
	//    nil - for NULL values
	//
	// An error should be returned if the value can not be stored
	// without loss of information.
	Scan(src interface{}) error
}

// ErrNoRows is returned by Scan when QueryRow doesn't return a
// row. In such a case, QueryRow returns a placeholder *Row value that
// defers this error until a Scan.
var ErrNoRows = errors.New("sql: no rows in result set")

// DB is a database handle. It's safe for concurrent use by multiple
// goroutines.
//
// The sql package creates and frees connections automatically; it
// also maintains a free pool of idle connections. If the database has
// a concept of per-connection state, such state can only be reliably
// observed within a transaction. Once DB.Begin is called, the
// returned Tx is bound to a single connection. Once Commit or
// Rollback is called on the transaction, that transaction's
// connection is returned to DB's idle connection pool. The pool size
// can be controlled with SetMaxIdleConns.
type DB struct {
	driver driver.Driver
	dsn    string

	mu       sync.Mutex // protects following fields
	freeConn []*driverConn
	closed   bool
	dep      map[finalCloser]depSet
	lastPut  map[*driverConn]string // stacktrace of last conn's put; debug only
	maxIdle  int                    // zero means defaultMaxIdleConns; negative means 0
}

// driverConn wraps a driver.Conn with a mutex, to
// be held during all calls into the Conn. (including any calls onto
// interfaces returned via that Conn, such as calls on Tx, Stmt,
// Result, Rows)
type driverConn struct {
	db *DB

	sync.Mutex  // guards following
	ci          driver.Conn
	closed      bool
	finalClosed bool // ci.Close has been called
	openStmt    map[driver.Stmt]bool

	// guarded by db.mu
	inUse      bool
	onPut      []func() // code (with db.mu held) run when conn is next returned
	dbmuClosed bool     // same as closed, but guarded by db.mu, for connIfFree
}

func (dc *driverConn) removeOpenStmt(si driver.Stmt) {
	dc.Lock()
	defer dc.Unlock()
	delete(dc.openStmt, si)
}

func (dc *driverConn) prepareLocked(query string) (driver.Stmt, error) {
	si, err := dc.ci.Prepare(query)
	if err == nil {
		// Track each driverConn's open statements, so we can close them
		// before closing the conn.
		//
		// TODO(bradfitz): let drivers opt out of caring about
		// stmt closes if the conn is about to close anyway? For now
		// do the safe thing, in case stmts need to be closed.
		//
		// TODO(bradfitz): after Go 1.1, closing driver.Stmts
		// should be moved to driverStmt, using unique
		// *driverStmts everywhere (including from
		// *Stmt.connStmt, instead of returning a
		// driver.Stmt), using driverStmt as a pointer
		// everywhere, and making it a finalCloser.
		if dc.openStmt == nil {
			dc.openStmt = make(map[driver.Stmt]bool)
		}
		dc.openStmt[si] = true
	}
	return si, err
}

// the dc.db's Mutex is held.
func (dc *driverConn) closeDBLocked() error {
	dc.Lock()
	if dc.closed {
		dc.Unlock()
		return errors.New("sql: duplicate driverConn close")
	}
	dc.closed = true
	dc.Unlock() // not defer; removeDep finalClose calls may need to lock
	return dc.db.removeDepLocked(dc, dc)()
}

func (dc *driverConn) Close() error {
	dc.Lock()
	if dc.closed {
		dc.Unlock()
		return errors.New("sql: duplicate driverConn close")
	}
	dc.closed = true
	dc.Unlock() // not defer; removeDep finalClose calls may need to lock

	// And now updates that require holding dc.mu.Lock.
	dc.db.mu.Lock()
	dc.dbmuClosed = true
	fn := dc.db.removeDepLocked(dc, dc)
	dc.db.mu.Unlock()
	return fn()
}

func (dc *driverConn) finalClose() error {
	dc.Lock()

	for si := range dc.openStmt {
		si.Close()
	}
	dc.openStmt = nil

	err := dc.ci.Close()
	dc.ci = nil
	dc.finalClosed = true

	dc.Unlock()
	return err
}

// driverStmt associates a driver.Stmt with the
// *driverConn from which it came, so the driverConn's lock can be
// held during calls.
type driverStmt struct {
	sync.Locker // the *driverConn
	si          driver.Stmt
}

func (ds *driverStmt) Close() error {
	ds.Lock()
	defer ds.Unlock()
	return ds.si.Close()
}

// depSet is a finalCloser's outstanding dependencies
type depSet map[interface{}]bool // set of true bools

// The finalCloser interface is used by (*DB).addDep and related
// dependency reference counting.
type finalCloser interface {
	// finalClose is called when the reference count of an object
	// goes to zero. (*DB).mu is not held while calling it.
	finalClose() error
}

// addDep notes that x now depends on dep, and x's finalClose won't be
// called until all of x's dependencies are removed with removeDep.
func (db *DB) addDep(x finalCloser, dep interface{}) {
	//println(fmt.Sprintf("addDep(%T %p, %T %p)", x, x, dep, dep))
	db.mu.Lock()
	defer db.mu.Unlock()
	db.addDepLocked(x, dep)
}

func (db *DB) addDepLocked(x finalCloser, dep interface{}) {
	if db.dep == nil {
		db.dep = make(map[finalCloser]depSet)
	}
	xdep := db.dep[x]
	if xdep == nil {
		xdep = make(depSet)
		db.dep[x] = xdep
	}
	xdep[dep] = true
}

// removeDep notes that x no longer depends on dep.
// If x still has dependencies, nil is returned.
// If x no longer has any dependencies, its finalClose method will be
// called and its error value will be returned.
func (db *DB) removeDep(x finalCloser, dep interface{}) error {
	db.mu.Lock()
	fn := db.removeDepLocked(x, dep)
	db.mu.Unlock()
	return fn()
}

func (db *DB) removeDepLocked(x finalCloser, dep interface{}) func() error {
	//println(fmt.Sprintf("removeDep(%T %p, %T %p)", x, x, dep, dep))
	done := false

	xdep := db.dep[x]
	if xdep != nil {
		delete(xdep, dep)
		if len(xdep) == 0 {
			delete(db.dep, x)
			done = true
		}
	}

	if !done {
		return func() error { return nil }
	}
	return func() error {
		//println(fmt.Sprintf("calling final close on %T %v (%#v)", x, x, x))
		return x.finalClose()
	}
}

// Open opens a database specified by its database driver name and a
// driver-specific data source name, usually consisting of at least a
// database name and connection information.
//
// Most users will open a database via a driver-specific connection
// helper function that returns a *DB. No database drivers are included
// in the Go standard library. See http://golang.org/s/sqldrivers for
// a list of third-party drivers.
//
// Open may just validate its arguments without creating a connection
// to the database. To verify that the data source name is valid, call
// Ping.
func Open(driverName, dataSourceName string) (*DB, error) {
	driveri, ok := drivers[driverName]
	if !ok {
		return nil, fmt.Errorf("sql: unknown driver %q (forgotten import?)", driverName)
	}
	db := &DB{
		driver:  driveri,
		dsn:     dataSourceName,
		lastPut: make(map[*driverConn]string),
	}
	return db, nil
}

// Ping verifies a connection to the database is still alive,
// establishing a connection if necessary.
func (db *DB) Ping() error {
	// TODO(bradfitz): give drivers an optional hook to implement
	// this in a more efficient or more reliable way, if they
	// have one.
	dc, err := db.conn()
	if err != nil {
		return err
	}
	db.putConn(dc, nil)
	return nil
}

// Close closes the database, releasing any open resources.
func (db *DB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()
	var err error
	for _, dc := range db.freeConn {
		err1 := dc.closeDBLocked()
		if err1 != nil {
			err = err1
		}
	}
	db.freeConn = nil
	db.closed = true
	return err
}

const defaultMaxIdleConns = 2

func (db *DB) maxIdleConnsLocked() int {
	n := db.maxIdle
	switch {
	case n == 0:
		// TODO(bradfitz): ask driver, if supported, for its default preference
		return defaultMaxIdleConns
	case n < 0:
		return 0
	default:
		return n
	}
}

// SetMaxIdleConns sets the maximum number of connections in the idle
// connection pool.
//
// If n <= 0, no idle connections are retained.
func (db *DB) SetMaxIdleConns(n int) {
	db.mu.Lock()
	defer db.mu.Unlock()
	if n > 0 {
		db.maxIdle = n
	} else {
		// No idle connections.
		db.maxIdle = -1
	}
	for len(db.freeConn) > 0 && len(db.freeConn) > n {
		nfree := len(db.freeConn)
		dc := db.freeConn[nfree-1]
		db.freeConn[nfree-1] = nil
		db.freeConn = db.freeConn[:nfree-1]
		go dc.Close()
	}
}

// conn returns a newly-opened or cached *driverConn
func (db *DB) conn() (*driverConn, error) {
	db.mu.Lock()
	if db.closed {
		db.mu.Unlock()
		return nil, errors.New("sql: database is closed")
	}
	if n := len(db.freeConn); n > 0 {
		conn := db.freeConn[n-1]
		db.freeConn = db.freeConn[:n-1]
		conn.inUse = true
		db.mu.Unlock()
		return conn, nil
	}
	db.mu.Unlock()

	ci, err := db.driver.Open(db.dsn)
	if err != nil {
		return nil, err
	}
	dc := &driverConn{
		db: db,
		ci: ci,
	}
	db.mu.Lock()
	db.addDepLocked(dc, dc)
	dc.inUse = true
	db.mu.Unlock()
	return dc, nil
}

var (
	errConnClosed = errors.New("database/sql: internal sentinel error: conn is closed")
	errConnBusy   = errors.New("database/sql: internal sentinel error: conn is busy")
)

// connIfFree returns (wanted, nil) if wanted is still a valid conn and
// isn't in use.
//
// The error is errConnClosed if the connection if the requested connection
// is invalid because it's been closed.
//
// The error is errConnBusy if the connection is in use.
func (db *DB) connIfFree(wanted *driverConn) (*driverConn, error) {
	db.mu.Lock()
	defer db.mu.Unlock()
	if wanted.inUse {
		return nil, errConnBusy
	}
	if wanted.dbmuClosed {
		return nil, errConnClosed
	}
	for i, conn := range db.freeConn {
		if conn != wanted {
			continue
		}
		db.freeConn[i] = db.freeConn[len(db.freeConn)-1]
		db.freeConn = db.freeConn[:len(db.freeConn)-1]
		wanted.inUse = true
		return wanted, nil
	}
	// TODO(bradfitz): shouldn't get here. After Go 1.1, change this to:
	// panic("connIfFree call requested a non-closed, non-busy, non-free conn")
	// Which passes all the tests, but I'm too paranoid to include this
	// late in Go 1.1.
	// Instead, treat it like a busy connection:
	return nil, errConnBusy
}

// putConnHook is a hook for testing.
var putConnHook func(*DB, *driverConn)

// noteUnusedDriverStatement notes that si is no longer used and should
// be closed whenever possible (when c is next not in use), unless c is
// already closed.
func (db *DB) noteUnusedDriverStatement(c *driverConn, si driver.Stmt) {
	db.mu.Lock()
	defer db.mu.Unlock()
	if c.inUse {
		c.onPut = append(c.onPut, func() {
			si.Close()
		})
	} else {
		c.Lock()
		defer c.Unlock()
		if !c.finalClosed {
			si.Close()
		}
	}
}

// debugGetPut determines whether getConn & putConn calls' stack traces
// are returned for more verbose crashes.
const debugGetPut = false

// putConn adds a connection to the db's free pool.
// err is optionally the last error that occurred on this connection.
func (db *DB) putConn(dc *driverConn, err error) {
	db.mu.Lock()
	if !dc.inUse {
		if debugGetPut {
			fmt.Printf("putConn(%v) DUPLICATE was: %s\n\nPREVIOUS was: %s", dc, stack(), db.lastPut[dc])
		}
		panic("sql: connection returned that was never out")
	}
	if debugGetPut {
		db.lastPut[dc] = stack()
	}
	dc.inUse = false

	for _, fn := range dc.onPut {
		fn()
	}
	dc.onPut = nil

	if err == driver.ErrBadConn {
		// Don't reuse bad connections.
		db.mu.Unlock()
		return
	}
	if putConnHook != nil {
		putConnHook(db, dc)
	}
	if n := len(db.freeConn); !db.closed && n < db.maxIdleConnsLocked() {
		db.freeConn = append(db.freeConn, dc)
		db.mu.Unlock()
		return
	}
	db.mu.Unlock()

	dc.Close()
}

// Prepare creates a prepared statement for later queries or executions.
// Multiple queries or executions may be run concurrently from the
// returned statement.
func (db *DB) Prepare(query string) (*Stmt, error) {
	var stmt *Stmt
	var err error
	for i := 0; i < 10; i++ {
		stmt, err = db.prepare(query)
		if err != driver.ErrBadConn {
			break
		}
	}
	return stmt, err
}

func (db *DB) prepare(query string) (*Stmt, error) {
	// TODO: check if db.driver supports an optional
	// driver.Preparer interface and call that instead, if so,
	// otherwise we make a prepared statement that's bound
	// to a connection, and to execute this prepared statement
	// we either need to use this connection (if it's free), else
	// get a new connection + re-prepare + execute on that one.
	dc, err := db.conn()
	if err != nil {
		return nil, err
	}
	dc.Lock()
	si, err := dc.prepareLocked(query)
	dc.Unlock()
	if err != nil {
		db.putConn(dc, err)
		return nil, err
	}
	stmt := &Stmt{
		db:    db,
		query: query,
		css:   []connStmt{{dc, si}},
	}
	db.addDep(stmt, stmt)
	db.putConn(dc, nil)
	return stmt, nil
}

// Exec executes a query without returning any rows.
// The args are for any placeholder parameters in the query.
func (db *DB) Exec(query string, args ...interface{}) (Result, error) {
	var res Result
	var err error
	for i := 0; i < 10; i++ {
		res, err = db.exec(query, args)
		if err != driver.ErrBadConn {
			break
		}
	}
	return res, err
}

func (db *DB) exec(query string, args []interface{}) (res Result, err error) {
	dc, err := db.conn()
	if err != nil {
		return nil, err
	}
	defer func() {
		db.putConn(dc, err)
	}()

	if execer, ok := dc.ci.(driver.Execer); ok {
		dargs, err := driverArgs(nil, args)
		if err != nil {
			return nil, err
		}
		dc.Lock()
		resi, err := execer.Exec(query, dargs)
		dc.Unlock()
		if err != driver.ErrSkip {
			if err != nil {
				return nil, err
			}
			return driverResult{dc, resi}, nil
		}
	}

	dc.Lock()
	si, err := dc.ci.Prepare(query)
	dc.Unlock()
	if err != nil {
		return nil, err
	}
	defer withLock(dc, func() { si.Close() })
	return resultFromStatement(driverStmt{dc, si}, args...)
}

// Query executes a query that returns rows, typically a SELECT.
// The args are for any placeholder parameters in the query.
func (db *DB) Query(query string, args ...interface{}) (*Rows, error) {
	var rows *Rows
	var err error
	for i := 0; i < 10; i++ {
		rows, err = db.query(query, args)
		if err != driver.ErrBadConn {
			break
		}
	}
	return rows, err
}

func (db *DB) query(query string, args []interface{}) (*Rows, error) {
	ci, err := db.conn()
	if err != nil {
		return nil, err
	}

	releaseConn := func(err error) { db.putConn(ci, err) }

	return db.queryConn(ci, releaseConn, query, args)
}

// queryConn executes a query on the given connection.
// The connection gets released by the releaseConn function.
func (db *DB) queryConn(dc *driverConn, releaseConn func(error), query string, args []interface{}) (*Rows, error) {
	if queryer, ok := dc.ci.(driver.Queryer); ok {
		dargs, err := driverArgs(nil, args)
		if err != nil {
			releaseConn(err)
			return nil, err
		}
		dc.Lock()
		rowsi, err := queryer.Query(query, dargs)
		dc.Unlock()
		if err != driver.ErrSkip {
			if err != nil {
				releaseConn(err)
				return nil, err
			}
			// Note: ownership of dc passes to the *Rows, to be freed
			// with releaseConn.
			rows := &Rows{
				dc:          dc,
				releaseConn: releaseConn,
				rowsi:       rowsi,
			}
			return rows, nil
		}
	}

	dc.Lock()
	si, err := dc.ci.Prepare(query)
	dc.Unlock()
	if err != nil {
		releaseConn(err)
		return nil, err
	}

	ds := driverStmt{dc, si}
	rowsi, err := rowsiFromStatement(ds, args...)
	if err != nil {
		releaseConn(err)
		dc.Lock()
		si.Close()
		dc.Unlock()
		return nil, err
	}

	// Note: ownership of ci passes to the *Rows, to be freed
	// with releaseConn.
	rows := &Rows{
		dc:          dc,
		releaseConn: releaseConn,
		rowsi:       rowsi,
		closeStmt:   si,
	}
	return rows, nil
}

// QueryRow executes a query that is expected to return at most one row.
// QueryRow always return a non-nil value. Errors are deferred until
// Row's Scan method is called.
func (db *DB) QueryRow(query string, args ...interface{}) *Row {
	rows, err := db.Query(query, args...)
	return &Row{rows: rows, err: err}
}

// Begin starts a transaction. The isolation level is dependent on
// the driver.
func (db *DB) Begin() (*Tx, error) {
	var tx *Tx
	var err error
	for i := 0; i < 10; i++ {
		tx, err = db.begin()
		if err != driver.ErrBadConn {
			break
		}
	}
	return tx, err
}

func (db *DB) begin() (tx *Tx, err error) {
	dc, err := db.conn()
	if err != nil {
		return nil, err
	}
	dc.Lock()
	txi, err := dc.ci.Begin()
	dc.Unlock()
	if err != nil {
		db.putConn(dc, err)
		return nil, err
	}
	return &Tx{
		db:  db,
		dc:  dc,
		txi: txi,
	}, nil
}

// Driver returns the database's underlying driver.
func (db *DB) Driver() driver.Driver {
	return db.driver
}

// Tx is an in-progress database transaction.
//
// A transaction must end with a call to Commit or Rollback.
//
// After a call to Commit or Rollback, all operations on the
// transaction fail with ErrTxDone.
type Tx struct {
	db *DB

	// dc is owned exclusively until Commit or Rollback, at which point
	// it's returned with putConn.
	dc  *driverConn
	txi driver.Tx

	// done transitions from false to true exactly once, on Commit
	// or Rollback. once done, all operations fail with
	// ErrTxDone.
	done bool
}

var ErrTxDone = errors.New("sql: Transaction has already been committed or rolled back")

func (tx *Tx) close() {
	if tx.done {
		panic("double close") // internal error
	}
	tx.done = true
	tx.db.putConn(tx.dc, nil)
	tx.dc = nil
	tx.txi = nil
}

func (tx *Tx) grabConn() (*driverConn, error) {
	if tx.done {
		return nil, ErrTxDone
	}
	return tx.dc, nil
}

// Commit commits the transaction.
func (tx *Tx) Commit() error {
	if tx.done {
		return ErrTxDone
	}
	defer tx.close()
	tx.dc.Lock()
	defer tx.dc.Unlock()
	return tx.txi.Commit()
}

// Rollback aborts the transaction.
func (tx *Tx) Rollback() error {
	if tx.done {
		return ErrTxDone
	}
	defer tx.close()
	tx.dc.Lock()
	defer tx.dc.Unlock()
	return tx.txi.Rollback()
}

// Prepare creates a prepared statement for use within a transaction.
//
// The returned statement operates within the transaction and can no longer
// be used once the transaction has been committed or rolled back.
//
// To use an existing prepared statement on this transaction, see Tx.Stmt.
func (tx *Tx) Prepare(query string) (*Stmt, error) {
	// TODO(bradfitz): We could be more efficient here and either
	// provide a method to take an existing Stmt (created on
	// perhaps a different Conn), and re-create it on this Conn if
	// necessary. Or, better: keep a map in DB of query string to
	// Stmts, and have Stmt.Execute do the right thing and
	// re-prepare if the Conn in use doesn't have that prepared
	// statement.  But we'll want to avoid caching the statement
	// in the case where we only call conn.Prepare implicitly
	// (such as in db.Exec or tx.Exec), but the caller package
	// can't be holding a reference to the returned statement.
	// Perhaps just looking at the reference count (by noting
	// Stmt.Close) would be enough. We might also want a finalizer
	// on Stmt to drop the reference count.
	dc, err := tx.grabConn()
	if err != nil {
		return nil, err
	}

	dc.Lock()
	si, err := dc.ci.Prepare(query)
	dc.Unlock()
	if err != nil {
		return nil, err
	}

	stmt := &Stmt{
		db: tx.db,
		tx: tx,
		txsi: &driverStmt{
			Locker: dc,
			si:     si,
		},
		query: query,
	}
	return stmt, nil
}

// Stmt returns a transaction-specific prepared statement from
// an existing statement.
//
// Example:
//  updateMoney, err := db.Prepare("UPDATE balance SET money=money+? WHERE id=?")
//  ...
//  tx, err := db.Begin()
//  ...
//  res, err := tx.Stmt(updateMoney).Exec(123.45, 98293203)
func (tx *Tx) Stmt(stmt *Stmt) *Stmt {
	// TODO(bradfitz): optimize this. Currently this re-prepares
	// each time.  This is fine for now to illustrate the API but
	// we should really cache already-prepared statements
	// per-Conn. See also the big comment in Tx.Prepare.

	if tx.db != stmt.db {
		return &Stmt{stickyErr: errors.New("sql: Tx.Stmt: statement from different database used")}
	}
	dc, err := tx.grabConn()
	if err != nil {
		return &Stmt{stickyErr: err}
	}
	dc.Lock()
	si, err := dc.ci.Prepare(stmt.query)
	dc.Unlock()
	return &Stmt{
		db: tx.db,
		tx: tx,
		txsi: &driverStmt{
			Locker: dc,
			si:     si,
		},
		query:     stmt.query,
		stickyErr: err,
	}
}

// Exec executes a query that doesn't return rows.
// For example: an INSERT and UPDATE.
func (tx *Tx) Exec(query string, args ...interface{}) (Result, error) {
	dc, err := tx.grabConn()
	if err != nil {
		return nil, err
	}

	if execer, ok := dc.ci.(driver.Execer); ok {
		dargs, err := driverArgs(nil, args)
		if err != nil {
			return nil, err
		}
		dc.Lock()
		resi, err := execer.Exec(query, dargs)
		dc.Unlock()
		if err == nil {
			return driverResult{dc, resi}, nil
		}
		if err != driver.ErrSkip {
			return nil, err
		}
	}

	dc.Lock()
	si, err := dc.ci.Prepare(query)
	dc.Unlock()
	if err != nil {
		return nil, err
	}
	defer withLock(dc, func() { si.Close() })

	return resultFromStatement(driverStmt{dc, si}, args...)
}

// Query executes a query that returns rows, typically a SELECT.
func (tx *Tx) Query(query string, args ...interface{}) (*Rows, error) {
	dc, err := tx.grabConn()
	if err != nil {
		return nil, err
	}
	releaseConn := func(error) {}
	return tx.db.queryConn(dc, releaseConn, query, args)
}

// QueryRow executes a query that is expected to return at most one row.
// QueryRow always return a non-nil value. Errors are deferred until
// Row's Scan method is called.
func (tx *Tx) QueryRow(query string, args ...interface{}) *Row {
	rows, err := tx.Query(query, args...)
	return &Row{rows: rows, err: err}
}

// connStmt is a prepared statement on a particular connection.
type connStmt struct {
	dc *driverConn
	si driver.Stmt
}

// Stmt is a prepared statement. Stmt is safe for concurrent use by multiple goroutines.
type Stmt struct {
	// Immutable:
	db        *DB    // where we came from
	query     string // that created the Stmt
	stickyErr error  // if non-nil, this error is returned for all operations

	closemu sync.RWMutex // held exclusively during close, for read otherwise.

	// If in a transaction, else both nil:
	tx   *Tx
	txsi *driverStmt

	mu     sync.Mutex // protects the rest of the fields
	closed bool

	// css is a list of underlying driver statement interfaces
	// that are valid on particular connections.  This is only
	// used if tx == nil and one is found that has idle
	// connections.  If tx != nil, txsi is always used.
	css []connStmt
}

// Exec executes a prepared statement with the given arguments and
// returns a Result summarizing the effect of the statement.
func (s *Stmt) Exec(args ...interface{}) (Result, error) {
	s.closemu.RLock()
	defer s.closemu.RUnlock()
	dc, releaseConn, si, err := s.connStmt()
	if err != nil {
		return nil, err
	}
	defer releaseConn(nil)

	return resultFromStatement(driverStmt{dc, si}, args...)
}

func resultFromStatement(ds driverStmt, args ...interface{}) (Result, error) {
	ds.Lock()
	want := ds.si.NumInput()
	ds.Unlock()

	// -1 means the driver doesn't know how to count the number of
	// placeholders, so we won't sanity check input here and instead let the
	// driver deal with errors.
	if want != -1 && len(args) != want {
		return nil, fmt.Errorf("sql: expected %d arguments, got %d", want, len(args))
	}

	dargs, err := driverArgs(&ds, args)
	if err != nil {
		return nil, err
	}

	ds.Lock()
	resi, err := ds.si.Exec(dargs)
	ds.Unlock()
	if err != nil {
		return nil, err
	}
	return driverResult{ds.Locker, resi}, nil
}

// connStmt returns a free driver connection on which to execute the
// statement, a function to call to release the connection, and a
// statement bound to that connection.
func (s *Stmt) connStmt() (ci *driverConn, releaseConn func(error), si driver.Stmt, err error) {
	if err = s.stickyErr; err != nil {
		return
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		err = errors.New("sql: statement is closed")
		return
	}

	// In a transaction, we always use the connection that the
	// transaction was created on.
	if s.tx != nil {
		s.mu.Unlock()
		ci, err = s.tx.grabConn() // blocks, waiting for the connection.
		if err != nil {
			return
		}
		releaseConn = func(error) {}
		return ci, releaseConn, s.txsi.si, nil
	}

	var cs connStmt
	match := false
	for i := 0; i < len(s.css); i++ {
		v := s.css[i]
		_, err := s.db.connIfFree(v.dc)
		if err == nil {
			match = true
			cs = v
			break
		}
		if err == errConnClosed {
			// Lazily remove dead conn from our freelist.
			s.css[i] = s.css[len(s.css)-1]
			s.css = s.css[:len(s.css)-1]
			i--
		}

	}
	s.mu.Unlock()

	// Make a new conn if all are busy.
	// TODO(bradfitz): or wait for one? make configurable later?
	if !match {
		for i := 0; ; i++ {
			dc, err := s.db.conn()
			if err != nil {
				return nil, nil, nil, err
			}
			dc.Lock()
			si, err := dc.prepareLocked(s.query)
			dc.Unlock()
			if err == driver.ErrBadConn && i < 10 {
				continue
			}
			if err != nil {
				return nil, nil, nil, err
			}
			s.mu.Lock()
			cs = connStmt{dc, si}
			s.css = append(s.css, cs)
			s.mu.Unlock()
			break
		}
	}

	conn := cs.dc
	releaseConn = func(err error) { s.db.putConn(conn, err) }
	return conn, releaseConn, cs.si, nil
}

// Query executes a prepared query statement with the given arguments
// and returns the query results as a *Rows.
func (s *Stmt) Query(args ...interface{}) (*Rows, error) {
	s.closemu.RLock()
	defer s.closemu.RUnlock()

	dc, releaseConn, si, err := s.connStmt()
	if err != nil {
		return nil, err
	}

	ds := driverStmt{dc, si}
	rowsi, err := rowsiFromStatement(ds, args...)
	if err != nil {
		releaseConn(err)
		return nil, err
	}

	// Note: ownership of ci passes to the *Rows, to be freed
	// with releaseConn.
	rows := &Rows{
		dc:    dc,
		rowsi: rowsi,
		// releaseConn set below
	}
	s.db.addDep(s, rows)
	rows.releaseConn = func(err error) {
		releaseConn(err)
		s.db.removeDep(s, rows)
	}
	return rows, nil
}

func rowsiFromStatement(ds driverStmt, args ...interface{}) (driver.Rows, error) {
	ds.Lock()
	want := ds.si.NumInput()
	ds.Unlock()

	// -1 means the driver doesn't know how to count the number of
	// placeholders, so we won't sanity check input here and instead let the
	// driver deal with errors.
	if want != -1 && len(args) != want {
		return nil, fmt.Errorf("sql: statement expects %d inputs; got %d", want, len(args))
	}

	dargs, err := driverArgs(&ds, args)
	if err != nil {
		return nil, err
	}

	ds.Lock()
	rowsi, err := ds.si.Query(dargs)
	ds.Unlock()
	if err != nil {
		return nil, err
	}
	return rowsi, nil
}

// QueryRow executes a prepared query statement with the given arguments.
// If an error occurs during the execution of the statement, that error will
// be returned by a call to Scan on the returned *Row, which is always non-nil.
// If the query selects no rows, the *Row's Scan will return ErrNoRows.
// Otherwise, the *Row's Scan scans the first selected row and discards
// the rest.
//
// Example usage:
//
//  var name string
//  err := nameByUseridStmt.QueryRow(id).Scan(&name)
func (s *Stmt) QueryRow(args ...interface{}) *Row {
	rows, err := s.Query(args...)
	if err != nil {
		return &Row{err: err}
	}
	return &Row{rows: rows}
}

// Close closes the statement.
func (s *Stmt) Close() error {
	s.closemu.Lock()
	defer s.closemu.Unlock()

	if s.stickyErr != nil {
		return s.stickyErr
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true

	if s.tx != nil {
		s.txsi.Close()
		return nil
	}

	return s.db.removeDep(s, s)
}

func (s *Stmt) finalClose() error {
	for _, v := range s.css {
		s.db.noteUnusedDriverStatement(v.dc, v.si)
		v.dc.removeOpenStmt(v.si)
		s.db.removeDep(v.dc, s)
	}
	s.css = nil
	return nil
}

// Rows is the result of a query. Its cursor starts before the first row
// of the result set. Use Next to advance through the rows:
//
//     rows, err := db.Query("SELECT ...")
//     ...
//     for rows.Next() {
//         var id int
//         var name string
//         err = rows.Scan(&id, &name)
//         ...
//     }
//     err = rows.Err() // get any error encountered during iteration
//     ...
type Rows struct {
	dc          *driverConn // owned; must call releaseConn when closed to release
	releaseConn func(error)
	rowsi       driver.Rows

	closed    bool
	lastcols  []driver.Value
	lasterr   error
	closeStmt driver.Stmt // if non-nil, statement to Close on close
}

// Next prepares the next result row for reading with the Scan method.
// It returns true on success, false if there is no next result row.
// Every call to Scan, even the first one, must be preceded by a call
// to Next.
func (rs *Rows) Next() bool {
	if rs.closed {
		return false
	}
	if rs.lasterr != nil {
		return false
	}
	if rs.lastcols == nil {
		rs.lastcols = make([]driver.Value, len(rs.rowsi.Columns()))
	}
	rs.lasterr = rs.rowsi.Next(rs.lastcols)
	if rs.lasterr == io.EOF {
		rs.Close()
	}
	return rs.lasterr == nil
}

// Err returns the error, if any, that was encountered during iteration.
func (rs *Rows) Err() error {
	if rs.lasterr == io.EOF {
		return nil
	}
	return rs.lasterr
}

// Columns returns the column names.
// Columns returns an error if the rows are closed, or if the rows
// are from QueryRow and there was a deferred error.
func (rs *Rows) Columns() ([]string, error) {
	if rs.closed {
		return nil, errors.New("sql: Rows are closed")
	}
	if rs.rowsi == nil {
		return nil, errors.New("sql: no Rows available")
	}
	return rs.rowsi.Columns(), nil
}

// Scan copies the columns in the current row into the values pointed
// at by dest.
//
// If an argument has type *[]byte, Scan saves in that argument a copy
// of the corresponding data. The copy is owned by the caller and can
// be modified and held indefinitely. The copy can be avoided by using
// an argument of type *RawBytes instead; see the documentation for
// RawBytes for restrictions on its use.
//
// If an argument has type *interface{}, Scan copies the value
// provided by the underlying driver without conversion. If the value
// is of type []byte, a copy is made and the caller owns the result.
func (rs *Rows) Scan(dest ...interface{}) error {
	if rs.closed {
		return errors.New("sql: Rows closed")
	}
	if rs.lasterr != nil {
		return rs.lasterr
	}
	if rs.lastcols == nil {
		return errors.New("sql: Scan called without calling Next")
	}
	if len(dest) != len(rs.lastcols) {
		return fmt.Errorf("sql: expected %d destination arguments in Scan, not %d", len(rs.lastcols), len(dest))
	}
	for i, sv := range rs.lastcols {
		err := convertAssign(dest[i], sv)
		if err != nil {
			return fmt.Errorf("sql: Scan error on column index %d: %v", i, err)
		}
	}
	return nil
}

// Close closes the Rows, preventing further enumeration. If the
// end is encountered, the Rows are closed automatically. Close
// is idempotent.
func (rs *Rows) Close() error {
	if rs.closed {
		return nil
	}
	rs.closed = true
	err := rs.rowsi.Close()
	if rs.closeStmt != nil {
		rs.closeStmt.Close()
	}
	rs.releaseConn(err)
	return err
}

// Row is the result of calling QueryRow to select a single row.
type Row struct {
	// One of these two will be non-nil:
	err  error // deferred error for easy chaining
	rows *Rows
}

// Scan copies the columns from the matched row into the values
// pointed at by dest.  If more than one row matches the query,
// Scan uses the first row and discards the rest.  If no row matches
// the query, Scan returns ErrNoRows.
func (r *Row) Scan(dest ...interface{}) error {
	if r.err != nil {
		return r.err
	}

	// TODO(bradfitz): for now we need to defensively clone all
	// []byte that the driver returned (not permitting
	// *RawBytes in Rows.Scan), since we're about to close
	// the Rows in our defer, when we return from this function.
	// the contract with the driver.Next(...) interface is that it
	// can return slices into read-only temporary memory that's
	// only valid until the next Scan/Close.  But the TODO is that
	// for a lot of drivers, this copy will be unnecessary.  We
	// should provide an optional interface for drivers to
	// implement to say, "don't worry, the []bytes that I return
	// from Next will not be modified again." (for instance, if
	// they were obtained from the network anyway) But for now we
	// don't care.
	for _, dp := range dest {
		if _, ok := dp.(*RawBytes); ok {
			return errors.New("sql: RawBytes isn't allowed on Row.Scan")
		}
	}

	defer r.rows.Close()
	if !r.rows.Next() {
		return ErrNoRows
	}
	err := r.rows.Scan(dest...)
	if err != nil {
		return err
	}

	return nil
}

// A Result summarizes an executed SQL command.
type Result interface {
	LastInsertId() (int64, error)
	RowsAffected() (int64, error)
}

type driverResult struct {
	sync.Locker // the *driverConn
	resi        driver.Result
}

func (dr driverResult) LastInsertId() (int64, error) {
	dr.Lock()
	defer dr.Unlock()
	return dr.resi.LastInsertId()
}

func (dr driverResult) RowsAffected() (int64, error) {
	dr.Lock()
	defer dr.Unlock()
	return dr.resi.RowsAffected()
}

func stack() string {
	var buf [2 << 10]byte
	return string(buf[:runtime.Stack(buf[:], false)])
}

// withLock runs while holding lk.
func withLock(lk sync.Locker, fn func()) {
	lk.Lock()
	fn()
	lk.Unlock()
}
