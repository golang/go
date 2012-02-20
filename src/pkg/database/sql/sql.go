// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sql provides a generic interface around SQL (or SQL-like)
// databases.
package sql

import (
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
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
		return nil, fmt.Errorf("sql: unknown driver %q (forgotten import?)", driverName)
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
		db.mu.Unlock()
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
	rows, err := stmt.Query(args...)
	if err != nil {
		stmt.Close()
		return nil, err
	}
	rows.closeStmt = stmt
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

	// ci is owned exclusively until Commit or Rollback, at which point
	// it's returned with putConn.
	ci  driver.Conn
	txi driver.Tx

	// cimu is held while somebody is using ci (between grabConn
	// and releaseConn)
	cimu sync.Mutex

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
	tx.db.putConn(tx.ci)
	tx.ci = nil
	tx.txi = nil
}

func (tx *Tx) grabConn() (driver.Conn, error) {
	if tx.done {
		return nil, ErrTxDone
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
		return ErrTxDone
	}
	defer tx.close()
	return tx.txi.Commit()
}

// Rollback aborts the transaction.
func (tx *Tx) Rollback() error {
	if tx.done {
		return ErrTxDone
	}
	defer tx.close()
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
	ci, err := tx.grabConn()
	if err != nil {
		return &Stmt{stickyErr: err}
	}
	defer tx.releaseConn()
	si, err := ci.Prepare(stmt.query)
	return &Stmt{
		db:        tx.db,
		tx:        tx,
		txsi:      si,
		query:     stmt.query,
		stickyErr: err,
	}
}

// Exec executes a query that doesn't return rows.
// For example: an INSERT and UPDATE.
func (tx *Tx) Exec(query string, args ...interface{}) (Result, error) {
	ci, err := tx.grabConn()
	if err != nil {
		return nil, err
	}
	defer tx.releaseConn()

	sargs, err := subsetTypeArgs(args)
	if err != nil {
		return nil, err
	}

	if execer, ok := ci.(driver.Execer); ok {
		resi, err := execer.Exec(query, sargs)
		if err == nil {
			return result{resi}, nil
		}
		if err != driver.ErrSkip {
			return nil, err
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
func (tx *Tx) Query(query string, args ...interface{}) (*Rows, error) {
	if tx.done {
		return nil, ErrTxDone
	}
	stmt, err := tx.Prepare(query)
	if err != nil {
		return nil, err
	}
	rows, err := stmt.Query(args...)
	if err == nil {
		rows.closeStmt = stmt
	}
	return rows, err
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
	db        *DB    // where we came from
	query     string // that created the Stmt
	stickyErr error  // if non-nil, this error is returned for all operations

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
		return nil, fmt.Errorf("sql: expected %d arguments, got %d", want, len(args))
	}

	sargs := make([]driver.Value, len(args))

	// Convert args to subset types.
	if cc, ok := si.(driver.ColumnConverter); ok {
		for n, arg := range args {
			// First, see if the value itself knows how to convert
			// itself to a driver type.  For example, a NullString
			// struct changing into a string or nil.
			if svi, ok := arg.(driver.Valuer); ok {
				sv, err := svi.Value()
				if err != nil {
					return nil, fmt.Errorf("sql: argument index %d from Value: %v", n, err)
				}
				if !driver.IsValue(sv) {
					return nil, fmt.Errorf("sql: argument index %d: non-subset type %T returned from Value", n, sv)
				}
				arg = sv
			}

			// Second, ask the column to sanity check itself. For
			// example, drivers might use this to make sure that
			// an int64 values being inserted into a 16-bit
			// integer field is in range (before getting
			// truncated), or that a nil can't go into a NOT NULL
			// column before going across the network to get the
			// same error.
			sargs[n], err = cc.ColumnConverter(n).ConvertValue(arg)
			if err != nil {
				return nil, fmt.Errorf("sql: converting Exec argument #%d's type: %v", n, err)
			}
			if !driver.IsValue(sargs[n]) {
				return nil, fmt.Errorf("sql: driver ColumnConverter error converted %T to unsupported type %T",
					arg, sargs[n])
			}
		}
	} else {
		for n, arg := range args {
			sargs[n], err = driver.DefaultParameterConverter.ConvertValue(arg)
			if err != nil {
				return nil, fmt.Errorf("sql: converting Exec argument #%d's type: %v", n, err)
			}
		}
	}

	resi, err := si.Exec(sargs)
	if err != nil {
		return nil, err
	}
	return result{resi}, nil
}

// connStmt returns a free driver connection on which to execute the
// statement, a function to call to release the connection, and a
// statement bound to that connection.
func (s *Stmt) connStmt() (ci driver.Conn, releaseConn func(), si driver.Stmt, err error) {
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
		return nil, fmt.Errorf("sql: statement expects %d inputs; got %d", si.NumInput(), len(args))
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

	closed    bool
	lastcols  []driver.Value
	lasterr   error
	closeStmt *Stmt // if non-nil, statement to Close on close
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
	for _, dp := range dest {
		b, ok := dp.(*[]byte)
		if !ok {
			continue
		}
		if *b == nil {
			// If the []byte is now nil (for a NULL value),
			// don't fall through to below which would
			// turn it into a non-nil 0-length byte slice
			continue
		}
		if _, ok = dp.(*RawBytes); ok {
			continue
		}
		clone := make([]byte, len(*b))
		copy(clone, *b)
		*b = clone
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
	if rs.closeStmt != nil {
		rs.closeStmt.Close()
	}
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

type result struct {
	driver.Result
}
