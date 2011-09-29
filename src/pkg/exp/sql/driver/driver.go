// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package driver defines interfaces to be implemented by database
// drivers as used by package sql.
//
// Code simply using databases should use package sql.
//
// Drivers only need to be aware of a subset of Go's types.  The db package
// will convert all types into one of the following:
//
//   int64
//   float64
//   bool
//   nil
//   []byte
//   string   [*] everywhere except from Rows.Next.
//
package driver

import (
	"os"
)

// Driver is the interface that must be implemented by a database
// driver.
type Driver interface {
	// Open returns a new or cached connection to the database.
	// The name is a string in a driver-specific format.
	//
	// The returned connection is only used by one goroutine at a
	// time.
	Open(name string) (Conn, os.Error)
}

// Execer is an optional interface that may be implemented by a Driver
// or a Conn.
//
// If a Driver does not implement Execer, the sql package's DB.Exec
// method first obtains a free connection from its free pool or from
// the driver's Open method. Execer should only be implemented by
// drivers that can provide a more efficient implementation.
//
// If a Conn does not implement Execer, the db package's DB.Exec will
// first prepare a query, execute the statement, and then close the
// statement.
//
// All arguments are of a subset type as defined in the package docs.
type Execer interface {
	Exec(query string, args []interface{}) (Result, os.Error)
}

// Conn is a connection to a database. It is not used concurrently
// by multiple goroutines.
//
// Conn is assumed to be stateful.
type Conn interface {
	// Prepare returns a prepared statement, bound to this connection.
	Prepare(query string) (Stmt, os.Error)

	// Close invalidates and potentially stops any current
	// prepared statements and transactions, marking this
	// connection as no longer in use.  The driver may cache or
	// close its underlying connection to its database.
	Close() os.Error

	// Begin starts and returns a new transaction.
	Begin() (Tx, os.Error)
}

// Result is the result of a query execution.
type Result interface {
	// LastInsertId returns the database's auto-generated ID
	// after, for example, an INSERT into a table with primary
	// key.
	LastInsertId() (int64, os.Error)

	// RowsAffected returns the number of rows affected by the
	// query.
	RowsAffected() (int64, os.Error)
}

// Stmt is a prepared statement. It is bound to a Conn and not
// used by multiple goroutines concurrently.
type Stmt interface {
	// Close closes the statement.
	Close() os.Error

	// NumInput returns the number of placeholder parameters.
	NumInput() int

	// Exec executes a query that doesn't return rows, such
	// as an INSERT or UPDATE.  The args are all of a subset
	// type as defined above.
	Exec(args []interface{}) (Result, os.Error)

	// Exec executes a query that may return rows, such as a
	// SELECT.  The args of all of a subset type as defined above.
	Query(args []interface{}) (Rows, os.Error)
}

// ColumnConverter may be optionally implemented by Stmt if the
// the statement is aware of its own columns' types and can
// convert from any type to a driver subset type.
type ColumnConverter interface {
	// ColumnConverter returns a ValueConverter for the provided
	// column index.  If the type of a specific column isn't known
	// or shouldn't be handled specially, DefaultValueConverter
	// can be returned.
	ColumnConverter(idx int) ValueConverter
}

// Rows is an iterator over an executed query's results.
type Rows interface {
	// Columns returns the names of the columns. The number of
	// columns of the result is inferred from the length of the
	// slice.  If a particular column name isn't known, an empty
	// string should be returned for that entry.
	Columns() []string

	// Close closes the rows iterator.
	Close() os.Error

	// Next is called to populate the next row of data into
	// the provided slice. The provided slice will be the same
	// size as the Columns() are wide.
	//
	// The dest slice may be populated with only with values
	// of subset types defined above, but excluding string.
	// All string values must be converted to []byte.
	Next(dest []interface{}) os.Error
}

// Tx is a transaction.
type Tx interface {
	Commit() os.Error
	Rollback() os.Error
}

// RowsAffected implements Result for an INSERT or UPDATE operation
// which mutates a number of rows.
type RowsAffected int64

var _ Result = RowsAffected(0)

func (RowsAffected) LastInsertId() (int64, os.Error) {
	return 0, os.NewError("no LastInsertId available")
}

func (v RowsAffected) RowsAffected() (int64, os.Error) {
	return int64(v), nil
}

// DDLSuccess is a pre-defined Result for drivers to return when a DDL
// command succeeds.
var DDLSuccess ddlSuccess

type ddlSuccess struct{}

var _ Result = ddlSuccess{}

func (ddlSuccess) LastInsertId() (int64, os.Error) {
	return 0, os.NewError("no LastInsertId available after DDL statement")
}

func (ddlSuccess) RowsAffected() (int64, os.Error) {
	return 0, os.NewError("no RowsAffected available after DDL statement")
}
