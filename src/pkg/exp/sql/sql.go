// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sql provides a generic interface around SQL (or SQL-like)
// databases.
package sql

import (
	"fmt"
	"os"
	"runtime"
	"sync"

	"exp/sql/driver"
)

var drivers = make(map[string]driver.Driver)

// Register makes a database driver available by the provided name.
// If Register is called twice with the same name or if driver is nil,
// it panics.
func Register(name string, driver driver.Driver) {
	if driver == nil {
		panic("db: Register driver is nil")
	}
	if _, dup := drivers[name]; dup {
		panic("db: Register called twice for driver " + name)
	}
	drivers[name] = driver
}

// NullableString represents a string that may be null.
// NullableString implements the ScannerInto interface so
// it can be used as a scan destination:
//
//  var s NullableString
//  err := db.QueryRow("SELECT name FROM foo WHERE id=?", id).Scan(&s)
//  ...
//  if s.Valid {
//     // use s.String
//  } else {
//     // NULL value
//  }
//
// TODO(bradfitz): add other types.
type NullableString struct {
	String string
	Valid  bool // Valid is true if String is not NULL
}

// ScanInto implements the ScannerInto interface.
func (ms *NullableString) ScanInto(value interface{}) os.Error {
	if value == nil {
		ms.String, ms.Valid = "", false
		return nil
	}
	ms.Valid = true
	return convertAssign(&ms.String, value)
}

// ScannerInto is an interface used by Scan.
type ScannerInto interface {
	// ScanInto assigns a value from a database driver.
	//
	// The value will be of one of the following restricted
	// set of types:
	//
	//    int64
	//    float64
	//    bool
	//    []byte
	//    nil - for NULL values
	//
	// An error should be returned if the value can not be stored
	// without loss of information.
	ScanInto(value interface{}) os.Error
}

// ErrNoRows is returned by Scan when QueryRow doesn't return a
// row. In such a case, QueryRow returns a placeholder *Row value that
// defers this error until a Scan.
var ErrNoRows = os.NewError("db: no rows in result set")

// DB is a database handle. It's safe for concurrent use by multiple
// goroutines.
type DB struct {
	driver driver.Driver
	dsn    string

	mu       sync.Mutex
	freeConn []driver.Conn
}

// Open opens a database specified by its database driver name and a
// driver-specific data source name, usually consisting of at least a
// database name and connection information.
//
// Most users will open a database via a driver-specific connection
// helper function that returns a *DB.
func Open(driverName, dataSourceName string) (*DB, os.Error) {
	driver, ok := drivers[driverName]
	if !ok {
		return nil, fmt.Errorf("db: unknown driver %q (forgotten import?)", driverName)
	}
	return &DB{driver: driver, dsn: dataSourceName}, nil
}

func (db *DB) maxIdleConns() int {
	const defaultMaxIdleConns = 2
	// TODO(bradfitz): ask driver, if supported, for its default preference
	// TODO(bradfitz): let users override?
	return defaultMaxIdleConns
}

// conn returns a newly-opened or cached driver.Conn
func (db *DB) conn() (driver.Conn, os.Error) {
	db.mu.Lock()
	if n := len(db.freeConn); n > 0 {
		conn := db.freeConn[n-1]
		db.freeConn = db.freeConn[:n-1]
		db.mu.Unlock()
		return conn, nil
	}
	db.mu.Unlock()
	return db.driver.Open(db.dsn)
}

func (db *DB) connIfFree(wanted driver.Conn) (conn driver.Conn, ok bool) {
	db.mu.Lock()
	defer db.mu.Unlock()
	for n, conn := range db.freeConn {
		if conn == wanted {
			db.freeConn[n] = db.freeConn[len(db.freeConn)-1]
			db.freeConn = db.freeConn[:len(db.freeConn)-1]
			return wanted, true
		}
	}
	return nil, false
}

func (db *DB) putConn(c driver.Conn) {
	if n := len(db.freeConn); n < db.maxIdleConns() {
		db.freeConn = append(db.freeConn, c)
		return
	}
	db.closeConn(c)
}

func (db *DB) closeConn(c driver.Conn) {
	// TODO: check to see if we need this Conn for any prepared statements
	// that are active.
	c.Close()
}

// Prepare creates a prepared statement for later execution.
func (db *DB) Prepare(query string) (*Stmt, os.Error) {
	// TODO: check if db.driver supports an optional
	// driver.Preparer interface and call that instead, if so,
	// otherwise we make a prepared statement that's bound
	// to a connection, and to execute this prepared statement
	// we either need to use this connection (if it's free), else
	// get a new connection + re-prepare + execute on that one.
	ci, err := db.conn()
	if err != nil {
		return nil, err
	}
	defer db.putConn(ci)
	si, err := ci.Prepare(query)
	if err != nil {
		return nil, err
	}
	stmt := &Stmt{
		db:    db,
		query: query,
		css:   []connStmt{{ci, si}},
	}
	return stmt, nil
}

// Exec executes a query without returning any rows.
func (db *DB) Exec(query string, args ...interface{}) (Result, os.Error) {
	// Optional fast path, if the driver implements driver.Execer.
	if execer, ok := db.driver.(driver.Execer); ok {
		resi, err := execer.Exec(query, args)
		if err != nil {
			return nil, err
		}
		return result{resi}, nil
	}

	// If the driver does not implement driver.Execer, we need
	// a connection.
	conn, err := db.conn()
	if err != nil {
		return nil, err
	}
	defer db.putConn(conn)

	if execer, ok := conn.(driver.Execer); ok {
		resi, err := execer.Exec(query, args)
		if err != nil {
			return nil, err
		}
		return result{resi}, nil
	}

	sti, err := conn.Prepare(query)
	if err != nil {
		return nil, err
	}
	defer sti.Close()
	resi, err := sti.Exec(args)
	if err != nil {
		return nil, err
	}
	return result{resi}, nil
}

// Query executes a query that returns rows, typically a SELECT.
func (db *DB) Query(query string, args ...interface{}) (*Rows, os.Error) {
	stmt, err := db.Prepare(query)
	if err != nil {
		return nil, err
	}
	defer stmt.Close()
	return stmt.Query(args...)
}

// QueryRow executes a query that is expected to return at most one row.
// QueryRow always return a non-nil value. Errors are deferred until
// Row's Scan method is called.
func (db *DB) QueryRow(query string, args ...interface{}) *Row {
	rows, err := db.Query(query, args...)
	if err != nil {
		return &Row{err: err}
	}
	return &Row{rows: rows}
}

// Begin starts a transaction.  The isolation level is dependent on
// the driver.
func (db *DB) Begin() (*Tx, os.Error) {
	// TODO(bradfitz): add another method for beginning a transaction
	// at a specific isolation level.
	panic(todo())
}

// DriverDatabase returns the database's underlying driver.
func (db *DB) Driver() driver.Driver {
	return db.driver
}

// Tx is an in-progress database transaction.
type Tx struct {

}

// Commit commits the transaction.
func (tx *Tx) Commit() os.Error {
	panic(todo())
}

// Rollback aborts the transaction.
func (tx *Tx) Rollback() os.Error {
	panic(todo())
}

// Prepare creates a prepared statement.
func (tx *Tx) Prepare(query string) (*Stmt, os.Error) {
	panic(todo())
}

// Exec executes a query that doesn't return rows.
// For example: an INSERT and UPDATE.
func (tx *Tx) Exec(query string, args ...interface{}) {
	panic(todo())
}

// Query executes a query that returns rows, typically a SELECT.
func (tx *Tx) Query(query string, args ...interface{}) (*Rows, os.Error) {
	panic(todo())
}

// QueryRow executes a query that is expected to return at most one row.
// QueryRow always return a non-nil value. Errors are deferred until
// Row's Scan method is called.
func (tx *Tx) QueryRow(query string, args ...interface{}) *Row {
	panic(todo())
}

// connStmt is a prepared statement on a particular connection.
type connStmt struct {
	ci driver.Conn
	si driver.Stmt
}

// Stmt is a prepared statement. Stmt is safe for concurrent use by multiple goroutines.
type Stmt struct {
	// Immutable:
	db    *DB    // where we came from
	query string // that created the Sttm

	mu     sync.Mutex
	closed bool
	css    []connStmt // can use any that have idle connections
}

func todo() string {
	_, file, line, _ := runtime.Caller(1)
	return fmt.Sprintf("%s:%d: TODO: implement", file, line)
}

// Exec executes a prepared statement with the given arguments and
// returns a Result summarizing the effect of the statement.
func (s *Stmt) Exec(args ...interface{}) (Result, os.Error) {
	ci, si, err := s.connStmt()
	if err != nil {
		return nil, err
	}
	defer s.db.putConn(ci)

	if want := si.NumInput(); len(args) != want {
		return nil, fmt.Errorf("db: expected %d arguments, got %d", want, len(args))
	}

	// Convert args to subset types.
	if cc, ok := si.(driver.ColumnConverter); ok {
		for n, arg := range args {
			args[n], err = cc.ColumnConverter(n).ConvertValue(arg)
			if err != nil {
				return nil, fmt.Errorf("db: converting Exec argument #%d's type: %v", n, err)
			}
			if !driver.IsParameterSubsetType(args[n]) {
				return nil, fmt.Errorf("db: driver ColumnConverter error converted %T to unsupported type %T",
					arg, args[n])
			}
		}
	} else {
		for n, arg := range args {
			args[n], err = driver.DefaultParameterConverter.ConvertValue(arg)
			if err != nil {
				return nil, fmt.Errorf("db: converting Exec argument #%d's type: %v", n, err)
			}
		}
	}

	resi, err := si.Exec(args)
	if err != nil {
		return nil, err
	}
	return result{resi}, nil
}

func (s *Stmt) connStmt(args ...interface{}) (driver.Conn, driver.Stmt, os.Error) {
	s.mu.Lock()
	if s.closed {
		return nil, nil, os.NewError("db: statement is closed")
	}
	var cs connStmt
	match := false
	for _, v := range s.css {
		// TODO(bradfitz): lazily clean up entries in this
		// list with dead conns while enumerating
		if _, match = s.db.connIfFree(cs.ci); match {
			cs = v
			break
		}
	}
	s.mu.Unlock()

	// Make a new conn if all are busy.
	// TODO(bradfitz): or wait for one? make configurable later?
	if !match {
		ci, err := s.db.conn()
		if err != nil {
			return nil, nil, err
		}
		si, err := ci.Prepare(s.query)
		if err != nil {
			return nil, nil, err
		}
		s.mu.Lock()
		cs = connStmt{ci, si}
		s.css = append(s.css, cs)
		s.mu.Unlock()
	}

	return cs.ci, cs.si, nil
}

// Query executes a prepared query statement with the given arguments
// and returns the query results as a *Rows.
func (s *Stmt) Query(args ...interface{}) (*Rows, os.Error) {
	ci, si, err := s.connStmt(args...)
	if err != nil {
		return nil, err
	}
	if len(args) != si.NumInput() {
		return nil, fmt.Errorf("db: statement expects %d inputs; got %d", si.NumInput(), len(args))
	}
	rowsi, err := si.Query(args)
	if err != nil {
		s.db.putConn(ci)
		return nil, err
	}
	// Note: ownership of ci passes to the *Rows
	rows := &Rows{
		db:    s.db,
		ci:    ci,
		rowsi: rowsi,
	}
	return rows, nil
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
//  err := nameByUseridStmt.QueryRow(id).Scan(&s)
func (s *Stmt) QueryRow(args ...interface{}) *Row {
	rows, err := s.Query(args...)
	if err != nil {
		return &Row{err: err}
	}
	return &Row{rows: rows}
}

// Close closes the statement.
func (s *Stmt) Close() os.Error {
	s.mu.Lock()
	defer s.mu.Unlock() // TODO(bradfitz): move this unlock after 'closed = true'?
	if s.closed {
		return nil
	}
	s.closed = true
	for _, v := range s.css {
		if ci, match := s.db.connIfFree(v.ci); match {
			v.si.Close()
			s.db.putConn(ci)
		} else {
			// TODO(bradfitz): care that we can't close
			// this statement because the statement's
			// connection is in use?
		}
	}
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
//     err = rows.Error() // get any Error encountered during iteration
//     ...
type Rows struct {
	db    *DB
	ci    driver.Conn // owned; must be returned when Rows is closed
	rowsi driver.Rows

	closed   bool
	lastcols []interface{}
	lasterr  os.Error
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
		rs.lastcols = make([]interface{}, len(rs.rowsi.Columns()))
	}
	rs.lasterr = rs.rowsi.Next(rs.lastcols)
	return rs.lasterr == nil
}

// Error returns the error, if any, that was encountered during iteration.
func (rs *Rows) Error() os.Error {
	if rs.lasterr == os.EOF {
		return nil
	}
	return rs.lasterr
}

// Scan copies the columns in the current row into the values pointed
// at by dest. If dest contains pointers to []byte, the slices should
// not be modified and should only be considered valid until the next
// call to Next or Scan.
func (rs *Rows) Scan(dest ...interface{}) os.Error {
	if rs.closed {
		return os.NewError("db: Rows closed")
	}
	if rs.lasterr != nil {
		return rs.lasterr
	}
	if rs.lastcols == nil {
		return os.NewError("db: Scan called without calling Next")
	}
	if len(dest) != len(rs.lastcols) {
		return fmt.Errorf("db: expected %d destination arguments in Scan, not %d", len(rs.lastcols), len(dest))
	}
	for i, sv := range rs.lastcols {
		err := convertAssign(dest[i], sv)
		if err != nil {
			return fmt.Errorf("db: Scan error on column index %d: %v", i, err)
		}
	}
	return nil
}

// Close closes the Rows, preventing further enumeration. If the
// end is encountered, the Rows are closed automatically. Close
// is idempotent.
func (rs *Rows) Close() os.Error {
	if rs.closed {
		return nil
	}
	rs.closed = true
	err := rs.rowsi.Close()
	rs.db.putConn(rs.ci)
	return err
}

// Row is the result of calling QueryRow to select a single row.
type Row struct {
	// One of these two will be non-nil:
	err  os.Error // deferred error for easy chaining
	rows *Rows
}

// Scan copies the columns from the matched row into the values
// pointed at by dest.  If more than one row matches the query,
// Scan uses the first row and discards the rest.  If no row matches
// the query, Scan returns ErrNoRows.
//
// If dest contains pointers to []byte, the slices should not be
// modified and should only be considered valid until the next call to
// Next or Scan.
func (r *Row) Scan(dest ...interface{}) os.Error {
	if r.err != nil {
		return r.err
	}
	defer r.rows.Close()
	if !r.rows.Next() {
		return ErrNoRows
	}
	return r.rows.Scan(dest...)
}

// A Result summarizes an executed SQL command.
type Result interface {
	LastInsertId() (int64, os.Error)
	RowsAffected() (int64, os.Error)
}

type result struct {
	driver.Result
}
