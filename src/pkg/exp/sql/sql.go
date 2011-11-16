// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sql provides a generic interface around SQL (or SQL-like)
// databases.
package sql

import (
	"errors"
	"fmt"
	"io"
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
func (ms *NullableString) ScanInto(value interface{}) error {
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
	ScanInto(value interface{}) error
}

// ErrNoRows is returned by Scan when QueryRow doesn't return a
// row. In such a case, QueryRow returns a placeholder *Row value that
// defers this error until a Scan.
var ErrNoRows = errors.New("db: no rows in result set")

// DB is a database handle. It's safe for concurrent use by multiple
// goroutines.
type DB struct {
	driver driver.Driver
	dsn    string

	mu       sync.Mutex // protects freeConn and closed
	freeConn []driver.Conn
	closed   bool
}

// Open opens a database specified by its database driver name and a
// driver-specific data source name, usually consisting of at least a
// database name and connection information.
//
// Most users will open a database via a driver-specific connection
// helper function that returns a *DB.
func Open(driverName, dataSourceName string) (*DB, error) {
	driver, ok := drivers[driverName]
	if !ok {
		return nil, fmt.Errorf("db: unknown driver %q (forgotten import?)", driverName)
	}
	return &DB{driver: driver, dsn: dataSourceName}, nil
}

// Close closes the database, releasing any open resources.
func (db *DB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()
	var err error
	for _, c := range db.freeConn {
		err1 := c.Close()
		if err1 != nil {
			err = err1
		}
	}
	db.freeConn = nil
	db.closed = true
	return err
}

func (db *DB) maxIdleConns() int {
	const defaultMaxIdleConns = 2
	// TODO(bradfitz): ask driver, if supported, for its default preference
	// TODO(bradfitz): let users override?
	return defaultMaxIdleConns
}

// conn returns a newly-opened or cached driver.Conn
func (db *DB) conn() (driver.Conn, error) {
	db.mu.Lock()
	if db.closed {
		return nil, errors.New("sql: database is closed")
	}
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
	db.mu.Lock()
	defer db.mu.Unlock()
	if n := len(db.freeConn); !db.closed && n < db.maxIdleConns() {
		db.freeConn = append(db.freeConn, c)
		return
	}
	db.closeConn(c) // TODO(bradfitz): release lock before calling this?
}

func (db *DB) closeConn(c driver.Conn) {
	// TODO: check to see if we need this Conn for any prepared statements
	// that are active.
	c.Close()
}

// Prepare creates a prepared statement for later execution.
func (db *DB) Prepare(query string) (*Stmt, error) {
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
func (db *DB) Exec(query string, args ...interface{}) (Result, error) {
	sargs, err := subsetTypeArgs(args)
	if err != nil {
		return nil, err
	}

	ci, err := db.conn()
	if err != nil {
		return nil, err
	}
	defer db.putConn(ci)

	if execer, ok := ci.(driver.Execer); ok {
		resi, err := execer.Exec(query, sargs)
		if err != driver.ErrSkip {
			if err != nil {
				return nil, err
			}
			return result{resi}, nil
		}
	}

	sti, err := ci.Prepare(query)
	if err != nil {
		return nil, err
	}
	defer sti.Close()

	resi, err := sti.Exec(sargs)
	if err != nil {
		return nil, err
	}
	return result{resi}, nil
}

// Query executes a query that returns rows, typically a SELECT.
func (db *DB) Query(query string, args ...interface{}) (*Rows, error) {
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
	return &Row{rows: rows, err: err}
}

// Begin starts a transaction. The isolation level is dependent on
// the driver.
func (db *DB) Begin() (*Tx, error) {
	ci, err := db.conn()
	if err != nil {
		return nil, err
	}
	txi, err := ci.Begin()
	if err != nil {
		db.putConn(ci)
		return nil, fmt.Errorf("sql: failed to Begin transaction: %v", err)
	}
	return &Tx{
		db:  db,
		ci:  ci,
		txi: txi,
	}, nil
}

// DriverDatabase returns the database's underlying driver.
func (db *DB) Driver() driver.Driver {
	return db.driver
}

// Tx is an in-progress database transaction.
//
// A transaction must end with a call to Commit or Rollback.
//
// After a call to Commit or Rollback, all operations on the
// transaction fail with ErrTransactionFinished.
type Tx struct {
	db *DB

	// ci is owned exclusively until Commit or Rollback, at which point
	// it's returned with putConn.
	ci  driver.Conn
	txi driver.Tx

	// cimu is held while somebody is using ci (between grabConn
	// and releaseConn)
	cimu sync.Mutex

	// done transitions from false to true exactly once, on Commit
	// or Rollback. once done, all operations fail with
	// ErrTransactionFinished.
	done bool
}

var ErrTransactionFinished = errors.New("sql: Transaction has already been committed or rolled back")

func (tx *Tx) close() {
	if tx.done {
		panic("double close") // internal error
	}
	tx.done = true
	tx.db.putConn(tx.ci)
	tx.ci = nil
	tx.txi = nil
}

func (tx *Tx) grabConn() (driver.Conn, error) {
	if tx.done {
		return nil, ErrTransactionFinished
	}
	tx.cimu.Lock()
	return tx.ci, nil
}

func (tx *Tx) releaseConn() {
	tx.cimu.Unlock()
}

// Commit commits the transaction.
func (tx *Tx) Commit() error {
	if tx.done {
		return ErrTransactionFinished
	}
	defer tx.close()
	return tx.txi.Commit()
}

// Rollback aborts the transaction.
func (tx *Tx) Rollback() error {
	if tx.done {
		return ErrTransactionFinished
	}
	defer tx.close()
	return tx.txi.Rollback()
}

// Prepare creates a prepared statement.
//
// The statement is only valid within the scope of this transaction.
func (tx *Tx) Prepare(query string) (*Stmt, error) {
	// TODO(bradfitz): the restriction that the returned statement
	// is only valid for this Transaction is lame and negates a
	// lot of the benefit of prepared statements.  We could be
	// more efficient here and either provide a method to take an
	// existing Stmt (created on perhaps a different Conn), and
	// re-create it on this Conn if necessary. Or, better: keep a
	// map in DB of query string to Stmts, and have Stmt.Execute
	// do the right thing and re-prepare if the Conn in use
	// doesn't have that prepared statement.  But we'll want to
	// avoid caching the statement in the case where we only call
	// conn.Prepare implicitly (such as in db.Exec or tx.Exec),
	// but the caller package can't be holding a reference to the
	// returned statement.  Perhaps just looking at the reference
	// count (by noting Stmt.Close) would be enough. We might also
	// want a finalizer on Stmt to drop the reference count.
	ci, err := tx.grabConn()
	if err != nil {
		return nil, err
	}
	defer tx.releaseConn()

	si, err := ci.Prepare(query)
	if err != nil {
		return nil, err
	}

	stmt := &Stmt{
		db:    tx.db,
		tx:    tx,
		txsi:  si,
		query: query,
	}
	return stmt, nil
}

// Exec executes a query that doesn't return rows.
// For example: an INSERT and UPDATE.
func (tx *Tx) Exec(query string, args ...interface{}) (Result, error) {
	ci, err := tx.grabConn()
	if err != nil {
		return nil, err
	}
	defer tx.releaseConn()

	if execer, ok := ci.(driver.Execer); ok {
		resi, err := execer.Exec(query, args)
		if err != nil {
			return nil, err
		}
		return result{resi}, nil
	}

	sti, err := ci.Prepare(query)
	if err != nil {
		return nil, err
	}
	defer sti.Close()

	sargs, err := subsetTypeArgs(args)
	if err != nil {
		return nil, err
	}

	resi, err := sti.Exec(sargs)
	if err != nil {
		return nil, err
	}
	return result{resi}, nil
}

// Query executes a query that returns rows, typically a SELECT.
func (tx *Tx) Query(query string, args ...interface{}) (*Rows, error) {
	if tx.done {
		return nil, ErrTransactionFinished
	}
	stmt, err := tx.Prepare(query)
	if err != nil {
		return nil, err
	}
	defer stmt.Close()
	return stmt.Query(args...)
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
	ci driver.Conn
	si driver.Stmt
}

// Stmt is a prepared statement. Stmt is safe for concurrent use by multiple goroutines.
type Stmt struct {
	// Immutable:
	db    *DB    // where we came from
	query string // that created the Sttm

	// If in a transaction, else both nil:
	tx   *Tx
	txsi driver.Stmt

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
	_, releaseConn, si, err := s.connStmt()
	if err != nil {
		return nil, err
	}
	defer releaseConn()

	// -1 means the driver doesn't know how to count the number of
	// placeholders, so we won't sanity check input here and instead let the
	// driver deal with errors.
	if want := si.NumInput(); want != -1 && len(args) != want {
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

// connStmt returns a free driver connection on which to execute the
// statement, a function to call to release the connection, and a
// statement bound to that connection.
func (s *Stmt) connStmt() (ci driver.Conn, releaseConn func(), si driver.Stmt, err error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		err = errors.New("db: statement is closed")
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
		releaseConn = func() { s.tx.releaseConn() }
		return ci, releaseConn, s.txsi, nil
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
			return nil, nil, nil, err
		}
		si, err := ci.Prepare(s.query)
		if err != nil {
			return nil, nil, nil, err
		}
		s.mu.Lock()
		cs = connStmt{ci, si}
		s.css = append(s.css, cs)
		s.mu.Unlock()
	}

	conn := cs.ci
	releaseConn = func() { s.db.putConn(conn) }
	return conn, releaseConn, cs.si, nil
}

// Query executes a prepared query statement with the given arguments
// and returns the query results as a *Rows.
func (s *Stmt) Query(args ...interface{}) (*Rows, error) {
	ci, releaseConn, si, err := s.connStmt()
	if err != nil {
		return nil, err
	}

	// -1 means the driver doesn't know how to count the number of
	// placeholders, so we won't sanity check input here and instead let the
	// driver deal with errors.
	if want := si.NumInput(); want != -1 && len(args) != want {
		return nil, fmt.Errorf("db: statement expects %d inputs; got %d", si.NumInput(), len(args))
	}
	sargs, err := subsetTypeArgs(args)
	if err != nil {
		return nil, err
	}
	rowsi, err := si.Query(sargs)
	if err != nil {
		s.db.putConn(ci)
		return nil, err
	}
	// Note: ownership of ci passes to the *Rows, to be freed
	// with releaseConn.
	rows := &Rows{
		db:          s.db,
		ci:          ci,
		releaseConn: releaseConn,
		rowsi:       rowsi,
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
func (s *Stmt) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true

	if s.tx != nil {
		s.txsi.Close()
	} else {
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
//     err = rows.Err() // get any error encountered during iteration
//     ...
type Rows struct {
	db          *DB
	ci          driver.Conn // owned; must call putconn when closed to release
	releaseConn func()
	rowsi       driver.Rows

	closed   bool
	lastcols []interface{}
	lasterr  error
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

// Err returns the error, if any, that was encountered during iteration.
func (rs *Rows) Err() error {
	if rs.lasterr == io.EOF {
		return nil
	}
	return rs.lasterr
}

// Scan copies the columns in the current row into the values pointed
// at by dest. If dest contains pointers to []byte, the slices should
// not be modified and should only be considered valid until the next
// call to Next or Scan.
func (rs *Rows) Scan(dest ...interface{}) error {
	if rs.closed {
		return errors.New("db: Rows closed")
	}
	if rs.lasterr != nil {
		return rs.lasterr
	}
	if rs.lastcols == nil {
		return errors.New("db: Scan called without calling Next")
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
func (rs *Rows) Close() error {
	if rs.closed {
		return nil
	}
	rs.closed = true
	err := rs.rowsi.Close()
	rs.releaseConn()
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
//
// If dest contains pointers to []byte, the slices should not be
// modified and should only be considered valid until the next call to
// Next or Scan.
func (r *Row) Scan(dest ...interface{}) error {
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
	LastInsertId() (int64, error)
	RowsAffected() (int64, error)
}

type result struct {
	driver.Result
}
