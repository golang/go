// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package driver defines interfaces to be implemented by database
// drivers as used by package sql.
//
// Most code should use package sql.
package driver

import (
	"context"
	"errors"
)

// Value is a value that drivers must be able to handle.
// It is either nil or an instance of one of these types:
//
//   int64
//   float64
//   bool
//   []byte
//   string
//   time.Time
type Value interface{}

// NamedValue holds both the value name and value.
// The Ordinal is the position of the parameter starting from one and is always set.
// If the Name is not empty it should be used for the parameter identifier and
// not the ordinal position.
type NamedValue struct {
	Name    string
	Ordinal int
	Value   Value
}

// Driver is the interface that must be implemented by a database
// driver.
type Driver interface {
	// Open returns a new connection to the database.
	// The name is a string in a driver-specific format.
	//
	// Open may return a cached connection (one previously
	// closed), but doing so is unnecessary; the sql package
	// maintains a pool of idle connections for efficient re-use.
	//
	// The returned connection is only used by one goroutine at a
	// time.
	Open(name string) (Conn, error)
}

// ErrSkip may be returned by some optional interfaces' methods to
// indicate at runtime that the fast path is unavailable and the sql
// package should continue as if the optional interface was not
// implemented. ErrSkip is only supported where explicitly
// documented.
var ErrSkip = errors.New("driver: skip fast-path; continue as if unimplemented")

// ErrBadConn should be returned by a driver to signal to the sql
// package that a driver.Conn is in a bad state (such as the server
// having earlier closed the connection) and the sql package should
// retry on a new connection.
//
// To prevent duplicate operations, ErrBadConn should NOT be returned
// if there's a possibility that the database server might have
// performed the operation. Even if the server sends back an error,
// you shouldn't return ErrBadConn.
var ErrBadConn = errors.New("driver: bad connection")

// Execer is an optional interface that may be implemented by a Conn.
//
// If a Conn does not implement Execer, the sql package's DB.Exec will
// first prepare a query, execute the statement, and then close the
// statement.
//
// Exec may return ErrSkip.
type Execer interface {
	Exec(query string, args []Value) (Result, error)
}

// ExecerContext is like execer, but must honor the context timeout and return
// when the context is cancelled.
type ExecerContext interface {
	ExecContext(ctx context.Context, query string, args []NamedValue) (Result, error)
}

// Queryer is an optional interface that may be implemented by a Conn.
//
// If a Conn does not implement Queryer, the sql package's DB.Query will
// first prepare a query, execute the statement, and then close the
// statement.
//
// Query may return ErrSkip.
type Queryer interface {
	Query(query string, args []Value) (Rows, error)
}

// QueryerContext is like Queryer, but most honor the context timeout and return
// when the context is cancelled.
type QueryerContext interface {
	QueryContext(ctx context.Context, query string, args []NamedValue) (Rows, error)
}

// Conn is a connection to a database. It is not used concurrently
// by multiple goroutines.
//
// Conn is assumed to be stateful.
type Conn interface {
	// Prepare returns a prepared statement, bound to this connection.
	Prepare(query string) (Stmt, error)

	// Close invalidates and potentially stops any current
	// prepared statements and transactions, marking this
	// connection as no longer in use.
	//
	// Because the sql package maintains a free pool of
	// connections and only calls Close when there's a surplus of
	// idle connections, it shouldn't be necessary for drivers to
	// do their own connection caching.
	Close() error

	// Begin starts and returns a new transaction.
	Begin() (Tx, error)
}

// ConnPrepareContext enhances the Conn interface with context.
type ConnPrepareContext interface {
	// PrepareContext returns a prepared statement, bound to this connection.
	// context is for the preparation of the statement,
	// it must not store the context within the statement itself.
	PrepareContext(ctx context.Context, query string) (Stmt, error)
}

// ConnBeginContext enhances the Conn interface with context.
type ConnBeginContext interface {
	// BeginContext starts and returns a new transaction.
	// the provided context should be used to roll the transaction back
	// if it is cancelled. If there is an isolation level in context
	// that is not supported by the driver an error must be returned.
	BeginContext(ctx context.Context) (Tx, error)
}

// Result is the result of a query execution.
type Result interface {
	// LastInsertId returns the database's auto-generated ID
	// after, for example, an INSERT into a table with primary
	// key.
	LastInsertId() (int64, error)

	// RowsAffected returns the number of rows affected by the
	// query.
	RowsAffected() (int64, error)
}

// Stmt is a prepared statement. It is bound to a Conn and not
// used by multiple goroutines concurrently.
type Stmt interface {
	// Close closes the statement.
	//
	// As of Go 1.1, a Stmt will not be closed if it's in use
	// by any queries.
	Close() error

	// NumInput returns the number of placeholder parameters.
	//
	// If NumInput returns >= 0, the sql package will sanity check
	// argument counts from callers and return errors to the caller
	// before the statement's Exec or Query methods are called.
	//
	// NumInput may also return -1, if the driver doesn't know
	// its number of placeholders. In that case, the sql package
	// will not sanity check Exec or Query argument counts.
	NumInput() int

	// Exec executes a query that doesn't return rows, such
	// as an INSERT or UPDATE.
	Exec(args []Value) (Result, error)

	// Query executes a query that may return rows, such as a
	// SELECT.
	Query(args []Value) (Rows, error)
}

// StmtExecContext enhances the Stmt interface by providing Exec with context.
type StmtExecContext interface {
	// ExecContext must honor the context timeout and return when it is cancelled.
	ExecContext(ctx context.Context, args []NamedValue) (Result, error)
}

// StmtQueryContext enhances the Stmt interface by providing Query with context.
type StmtQueryContext interface {
	// QueryContext must honor the context timeout and return when it is cancelled.
	QueryContext(ctx context.Context, args []NamedValue) (Rows, error)
}

// ColumnConverter may be optionally implemented by Stmt if the
// statement is aware of its own columns' types and can convert from
// any type to a driver Value.
type ColumnConverter interface {
	// ColumnConverter returns a ValueConverter for the provided
	// column index. If the type of a specific column isn't known
	// or shouldn't be handled specially, DefaultValueConverter
	// can be returned.
	ColumnConverter(idx int) ValueConverter
}

// Rows is an iterator over an executed query's results.
type Rows interface {
	// Columns returns the names of the columns. The number of
	// columns of the result is inferred from the length of the
	// slice. If a particular column name isn't known, an empty
	// string should be returned for that entry.
	Columns() []string

	// Close closes the rows iterator.
	Close() error

	// Next is called to populate the next row of data into
	// the provided slice. The provided slice will be the same
	// size as the Columns() are wide.
	//
	// Next should return io.EOF when there are no more rows.
	Next(dest []Value) error
}

// RowsNextResultSet extends the Rows interface by providing a way to signal
// the driver to advance to the next result set.
type RowsNextResultSet interface {
	Rows

	// HasNextResultSet is called at the end of the current result set and
	// reports whether there is another result set after the current one.
	HasNextResultSet() bool

	// NextResultSet advances the driver to the next result set even
	// if there are remaining rows in the current result set.
	//
	// NextResultSet should return io.EOF when there are no more result sets.
	NextResultSet() error
}

// Tx is a transaction.
type Tx interface {
	Commit() error
	Rollback() error
}

// RowsAffected implements Result for an INSERT or UPDATE operation
// which mutates a number of rows.
type RowsAffected int64

var _ Result = RowsAffected(0)

func (RowsAffected) LastInsertId() (int64, error) {
	return 0, errors.New("no LastInsertId available")
}

func (v RowsAffected) RowsAffected() (int64, error) {
	return int64(v), nil
}

// ResultNoRows is a pre-defined Result for drivers to return when a DDL
// command (such as a CREATE TABLE) succeeds. It returns an error for both
// LastInsertId and RowsAffected.
var ResultNoRows noRows

type noRows struct{}

var _ Result = noRows{}

func (noRows) LastInsertId() (int64, error) {
	return 0, errors.New("no LastInsertId available after DDL statement")
}

func (noRows) RowsAffected() (int64, error) {
	return 0, errors.New("no RowsAffected available after DDL statement")
}
