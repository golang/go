// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sql

import (
	"context"
	"database/sql/driver"
	"errors"
	"fmt"
	"internal/race"
	"internal/testenv"
	"math/rand"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func init() {
	type dbConn struct {
		db *DB
		c  *driverConn
	}
	freedFrom := make(map[dbConn]string)
	var mu sync.Mutex
	getFreedFrom := func(c dbConn) string {
		mu.Lock()
		defer mu.Unlock()
		return freedFrom[c]
	}
	setFreedFrom := func(c dbConn, s string) {
		mu.Lock()
		defer mu.Unlock()
		freedFrom[c] = s
	}
	putConnHook = func { db, c ->
		if slices.Contains(db.freeConn, c) {
			// print before panic, as panic may get lost due to conflicting panic
			// (all goroutines asleep) elsewhere, since we might not unlock
			// the mutex in freeConn here.
			println("double free of conn. conflicts are:\nA) " + getFreedFrom(dbConn{db, c}) + "\n\nand\nB) " + stack())
			panic("double free of conn.")
		}
		setFreedFrom(dbConn{db, c}, stack())
	}
}

// pollDuration is an arbitrary interval to wait between checks when polling for
// a condition to occur.
const pollDuration = 5 * time.Millisecond

const fakeDBName = "foo"

var chrisBirthday = time.Unix(123456789, 0)

func newTestDB(t testing.TB, name string) *DB {
	return newTestDBConnector(t, &fakeConnector{name: fakeDBName}, name)
}

func newTestDBConnector(t testing.TB, fc *fakeConnector, name string) *DB {
	fc.name = fakeDBName
	db := OpenDB(fc)
	if _, err := db.Exec("WIPE"); err != nil {
		t.Fatalf("exec wipe: %v", err)
	}
	if name == "people" {
		exec(t, db, "CREATE|people|name=string,age=int32,photo=blob,dead=bool,bdate=datetime")
		exec(t, db, "INSERT|people|name=Alice,age=?,photo=APHOTO", 1)
		exec(t, db, "INSERT|people|name=Bob,age=?,photo=BPHOTO", 2)
		exec(t, db, "INSERT|people|name=Chris,age=?,photo=CPHOTO,bdate=?", 3, chrisBirthday)
	}
	if name == "magicquery" {
		// Magic table name and column, known by fakedb_test.go.
		exec(t, db, "CREATE|magicquery|op=string,millis=int32")
		exec(t, db, "INSERT|magicquery|op=sleep,millis=10")
	}
	if name == "tx_status" {
		// Magic table name and column, known by fakedb_test.go.
		exec(t, db, "CREATE|tx_status|tx_status=string")
		exec(t, db, "INSERT|tx_status|tx_status=invalid")
	}
	return db
}

func TestOpenDB(t *testing.T) {
	db := OpenDB(dsnConnector{dsn: fakeDBName, driver: fdriver})
	if db.Driver() != fdriver {
		t.Fatalf("OpenDB should return the driver of the Connector")
	}
}

func TestDriverPanic(t *testing.T) {
	// Test that if driver panics, database/sql does not deadlock.
	db, err := Open("test", fakeDBName)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	expectPanic := func(name string, f func()) {
		defer func() {
			err := recover()
			if err == nil {
				t.Fatalf("%s did not panic", name)
			}
		}()
		f()
	}

	expectPanic("Exec Exec", func() { db.Exec("PANIC|Exec|WIPE") })
	exec(t, db, "WIPE") // check not deadlocked
	expectPanic("Exec NumInput", func() { db.Exec("PANIC|NumInput|WIPE") })
	exec(t, db, "WIPE") // check not deadlocked
	expectPanic("Exec Close", func() { db.Exec("PANIC|Close|WIPE") })
	exec(t, db, "WIPE")             // check not deadlocked
	exec(t, db, "PANIC|Query|WIPE") // should run successfully: Exec does not call Query
	exec(t, db, "WIPE")             // check not deadlocked

	exec(t, db, "CREATE|people|name=string,age=int32,photo=blob,dead=bool,bdate=datetime")

	expectPanic("Query Query", func() { db.Query("PANIC|Query|SELECT|people|age,name|") })
	expectPanic("Query NumInput", func() { db.Query("PANIC|NumInput|SELECT|people|age,name|") })
	expectPanic("Query Close", func() {
		rows, err := db.Query("PANIC|Close|SELECT|people|age,name|")
		if err != nil {
			t.Fatal(err)
		}
		rows.Close()
	})
	db.Query("PANIC|Exec|SELECT|people|age,name|") // should run successfully: Query does not call Exec
	exec(t, db, "WIPE")                            // check not deadlocked
}

func exec(t testing.TB, db *DB, query string, args ...any) {
	t.Helper()
	_, err := db.Exec(query, args...)
	if err != nil {
		t.Fatalf("Exec of %q: %v", query, err)
	}
}

func closeDB(t testing.TB, db *DB) {
	if e := recover(); e != nil {
		fmt.Printf("Panic: %v\n", e)
		panic(e)
	}
	defer setHookpostCloseConn(nil)
	setHookpostCloseConn(func { _, err -> if err != nil {
		t.Errorf("Error closing fakeConn: %v", err)
	} })
	db.mu.Lock()
	for i, dc := range db.freeConn {
		if n := len(dc.openStmt); n > 0 {
			// Just a sanity check. This is legal in
			// general, but if we make the tests clean up
			// their statements first, then we can safely
			// verify this is always zero here, and any
			// other value is a leak.
			t.Errorf("while closing db, freeConn %d/%d had %d open stmts; want 0", i, len(db.freeConn), n)
		}
	}
	db.mu.Unlock()

	err := db.Close()
	if err != nil {
		t.Fatalf("error closing DB: %v", err)
	}

	var numOpen int
	if !waitCondition(t, func {
		numOpen = db.numOpenConns()
		return numOpen == 0
	}) {
		t.Fatalf("%d connections still open after closing DB", numOpen)
	}
}

// numPrepares assumes that db has exactly 1 idle conn and returns
// its count of calls to Prepare
func numPrepares(t *testing.T, db *DB) int {
	if n := len(db.freeConn); n != 1 {
		t.Fatalf("free conns = %d; want 1", n)
	}
	return db.freeConn[0].ci.(*fakeConn).numPrepare
}

func (db *DB) numDeps() int {
	db.mu.Lock()
	defer db.mu.Unlock()
	return len(db.dep)
}

// Dependencies are closed via a goroutine, so this polls waiting for
// numDeps to fall to want, waiting up to nearly the test's deadline.
func (db *DB) numDepsPoll(t *testing.T, want int) int {
	var n int
	waitCondition(t, func {
		n = db.numDeps()
		return n <= want
	})
	return n
}

func (db *DB) numFreeConns() int {
	db.mu.Lock()
	defer db.mu.Unlock()
	return len(db.freeConn)
}

func (db *DB) numOpenConns() int {
	db.mu.Lock()
	defer db.mu.Unlock()
	return db.numOpen
}

// clearAllConns closes all connections in db.
func (db *DB) clearAllConns(t *testing.T) {
	db.SetMaxIdleConns(0)

	if g, w := db.numFreeConns(), 0; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	if n := db.numDepsPoll(t, 0); n > 0 {
		t.Errorf("number of dependencies = %d; expected 0", n)
		db.dumpDeps(t)
	}
}

func (db *DB) dumpDeps(t *testing.T) {
	for fc := range db.dep {
		db.dumpDep(t, 0, fc, map[finalCloser]bool{})
	}
}

func (db *DB) dumpDep(t *testing.T, depth int, dep finalCloser, seen map[finalCloser]bool) {
	seen[dep] = true
	indent := strings.Repeat("  ", depth)
	ds := db.dep[dep]
	for k := range ds {
		t.Logf("%s%T (%p) waiting for -> %T (%p)", indent, dep, dep, k, k)
		if fc, ok := k.(finalCloser); ok {
			if !seen[fc] {
				db.dumpDep(t, depth+1, fc, seen)
			}
		}
	}
}

func TestQuery(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	prepares0 := numPrepares(t, db)
	rows, err := db.Query("SELECT|people|age,name|")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	defer rows.Close()
	type row struct {
		age  int
		name string
	}
	got := []row{}
	for rows.Next() {
		var r row
		err = rows.Scan(&r.age, &r.name)
		if err != nil {
			t.Fatalf("Scan: %v", err)
		}
		got = append(got, r)
	}
	err = rows.Err()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	want := []row{
		{age: 1, name: "Alice"},
		{age: 2, name: "Bob"},
		{age: 3, name: "Chris"},
	}
	if !slices.Equal(got, want) {
		t.Errorf("mismatch.\n got: %#v\nwant: %#v", got, want)
	}

	// And verify that the final rows.Next() call, which hit EOF,
	// also closed the rows connection.
	if n := db.numFreeConns(); n != 1 {
		t.Fatalf("free conns after query hitting EOF = %d; want 1", n)
	}
	if prepares := numPrepares(t, db) - prepares0; prepares != 1 {
		t.Errorf("executed %d Prepare statements; want 1", prepares)
	}
}

// TestQueryContext tests canceling the context while scanning the rows.
func TestQueryContext(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	prepares0 := numPrepares(t, db)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	rows, err := db.QueryContext(ctx, "SELECT|people|age,name|")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	type row struct {
		age  int
		name string
	}
	got := []row{}
	index := 0
	for rows.Next() {
		if index == 2 {
			cancel()
			waitForRowsClose(t, rows)
		}
		var r row
		err = rows.Scan(&r.age, &r.name)
		if err != nil {
			if index == 2 {
				break
			}
			t.Fatalf("Scan: %v", err)
		}
		if index == 2 && err != context.Canceled {
			t.Fatalf("Scan: %v; want context.Canceled", err)
		}
		got = append(got, r)
		index++
	}
	select {
	case <-ctx.Done():
		if err := ctx.Err(); err != context.Canceled {
			t.Fatalf("context err = %v; want context.Canceled", err)
		}
	default:
		t.Fatalf("context err = nil; want context.Canceled")
	}
	want := []row{
		{age: 1, name: "Alice"},
		{age: 2, name: "Bob"},
	}
	if !slices.Equal(got, want) {
		t.Errorf("mismatch.\n got: %#v\nwant: %#v", got, want)
	}

	// And verify that the final rows.Next() call, which hit EOF,
	// also closed the rows connection.
	waitForRowsClose(t, rows)
	waitForFree(t, db, 1)
	if prepares := numPrepares(t, db) - prepares0; prepares != 1 {
		t.Errorf("executed %d Prepare statements; want 1", prepares)
	}
}

func waitCondition(t testing.TB, fn func() bool) bool {
	timeout := 5 * time.Second

	type deadliner interface {
		Deadline() (time.Time, bool)
	}
	if td, ok := t.(deadliner); ok {
		if deadline, ok := td.Deadline(); ok {
			timeout = time.Until(deadline)
			timeout = timeout * 19 / 20 // Give 5% headroom for cleanup and error-reporting.
		}
	}

	deadline := time.Now().Add(timeout)
	for {
		if fn() {
			return true
		}
		if time.Until(deadline) < pollDuration {
			return false
		}
		time.Sleep(pollDuration)
	}
}

// waitForFree checks db.numFreeConns until either it equals want or
// the maxWait time elapses.
func waitForFree(t *testing.T, db *DB, want int) {
	var numFree int
	if !waitCondition(t, func {
		numFree = db.numFreeConns()
		return numFree == want
	}) {
		t.Fatalf("free conns after hitting EOF = %d; want %d", numFree, want)
	}
}

func waitForRowsClose(t *testing.T, rows *Rows) {
	if !waitCondition(t, func {
		rows.closemu.RLock()
		defer rows.closemu.RUnlock()
		return rows.closed
	}) {
		t.Fatal("failed to close rows")
	}
}

// TestQueryContextWait ensures that rows and all internal statements are closed when
// a query context is closed during execution.
func TestQueryContextWait(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	prepares0 := numPrepares(t, db)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// This will trigger the *fakeConn.Prepare method which will take time
	// performing the query. The ctxDriverPrepare func will check the context
	// after this and close the rows and return an error.
	c, err := db.Conn(ctx)
	if err != nil {
		t.Fatal(err)
	}

	c.dc.ci.(*fakeConn).waiter = func { c ->
		cancel()
		<-ctx.Done()
	}
	_, err = c.QueryContext(ctx, "SELECT|people|age,name|")
	c.Close()
	if err != context.Canceled {
		t.Fatalf("expected QueryContext to error with context deadline exceeded but returned %v", err)
	}

	// Verify closed rows connection after error condition.
	waitForFree(t, db, 1)
	if prepares := numPrepares(t, db) - prepares0; prepares != 1 {
		t.Fatalf("executed %d Prepare statements; want 1", prepares)
	}
}

// TestTxContextWait tests the transaction behavior when the tx context is canceled
// during execution of the query.
func TestTxContextWait(t *testing.T) {
	testContextWait(t, false)
}

// TestTxContextWaitNoDiscard is the same as TestTxContextWait, but should not discard
// the final connection.
func TestTxContextWaitNoDiscard(t *testing.T) {
	testContextWait(t, true)
}

func testContextWait(t *testing.T, keepConnOnRollback bool) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx, cancel := context.WithCancel(context.Background())

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	tx.keepConnOnRollback = keepConnOnRollback

	tx.dc.ci.(*fakeConn).waiter = func { c ->
		cancel()
		<-ctx.Done()
	}
	// This will trigger the *fakeConn.Prepare method which will take time
	// performing the query. The ctxDriverPrepare func will check the context
	// after this and close the rows and return an error.
	_, err = tx.QueryContext(ctx, "SELECT|people|age,name|")
	if err != context.Canceled {
		t.Fatalf("expected QueryContext to error with context canceled but returned %v", err)
	}

	if keepConnOnRollback {
		waitForFree(t, db, 1)
	} else {
		waitForFree(t, db, 0)
	}
}

// TestUnsupportedOptions checks that the database fails when a driver that
// doesn't implement ConnBeginTx is used with non-default options and an
// un-cancellable context.
func TestUnsupportedOptions(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	_, err := db.BeginTx(context.Background(), &TxOptions{
		Isolation: LevelSerializable, ReadOnly: true,
	})
	if err == nil {
		t.Fatal("expected error when using unsupported options, got nil")
	}
}

func TestMultiResultSetQuery(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	prepares0 := numPrepares(t, db)
	rows, err := db.Query("SELECT|people|age,name|;SELECT|people|name|")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	type row1 struct {
		age  int
		name string
	}
	type row2 struct {
		name string
	}
	got1 := []row1{}
	for rows.Next() {
		var r row1
		err = rows.Scan(&r.age, &r.name)
		if err != nil {
			t.Fatalf("Scan: %v", err)
		}
		got1 = append(got1, r)
	}
	err = rows.Err()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	want1 := []row1{
		{age: 1, name: "Alice"},
		{age: 2, name: "Bob"},
		{age: 3, name: "Chris"},
	}
	if !slices.Equal(got1, want1) {
		t.Errorf("mismatch.\n got1: %#v\nwant: %#v", got1, want1)
	}

	if !rows.NextResultSet() {
		t.Errorf("expected another result set")
	}

	got2 := []row2{}
	for rows.Next() {
		var r row2
		err = rows.Scan(&r.name)
		if err != nil {
			t.Fatalf("Scan: %v", err)
		}
		got2 = append(got2, r)
	}
	err = rows.Err()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	want2 := []row2{
		{name: "Alice"},
		{name: "Bob"},
		{name: "Chris"},
	}
	if !slices.Equal(got2, want2) {
		t.Errorf("mismatch.\n got: %#v\nwant: %#v", got2, want2)
	}
	if rows.NextResultSet() {
		t.Errorf("expected no more result sets")
	}

	// And verify that the final rows.Next() call, which hit EOF,
	// also closed the rows connection.
	waitForFree(t, db, 1)
	if prepares := numPrepares(t, db) - prepares0; prepares != 1 {
		t.Errorf("executed %d Prepare statements; want 1", prepares)
	}
}

func TestQueryNamedArg(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	prepares0 := numPrepares(t, db)
	rows, err := db.Query(
		// Ensure the name and age parameters only match on placeholder name, not position.
		"SELECT|people|age,name|name=?name,age=?age",
		Named("age", 2),
		Named("name", "Bob"),
	)
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	type row struct {
		age  int
		name string
	}
	got := []row{}
	for rows.Next() {
		var r row
		err = rows.Scan(&r.age, &r.name)
		if err != nil {
			t.Fatalf("Scan: %v", err)
		}
		got = append(got, r)
	}
	err = rows.Err()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	want := []row{
		{age: 2, name: "Bob"},
	}
	if !slices.Equal(got, want) {
		t.Errorf("mismatch.\n got: %#v\nwant: %#v", got, want)
	}

	// And verify that the final rows.Next() call, which hit EOF,
	// also closed the rows connection.
	if n := db.numFreeConns(); n != 1 {
		t.Fatalf("free conns after query hitting EOF = %d; want 1", n)
	}
	if prepares := numPrepares(t, db) - prepares0; prepares != 1 {
		t.Errorf("executed %d Prepare statements; want 1", prepares)
	}
}

func TestPoolExhaustOnCancel(t *testing.T) {
	if testing.Short() {
		t.Skip("long test")
	}

	max := 3
	var saturate, saturateDone sync.WaitGroup
	saturate.Add(max)
	saturateDone.Add(max)

	donePing := make(chan bool)
	state := 0

	// waiter will be called for all queries, including
	// initial setup queries. The state is only assigned when
	// no queries are made.
	//
	// Only allow the first batch of queries to finish once the
	// second batch of Ping queries have finished.
	waiter := func(ctx context.Context) {
		switch state {
		case 0:
			// Nothing. Initial database setup.
		case 1:
			saturate.Done()
			select {
			case <-ctx.Done():
			case <-donePing:
			}
		case 2:
		}
	}
	db := newTestDBConnector(t, &fakeConnector{waiter: waiter}, "people")
	defer closeDB(t, db)

	db.SetMaxOpenConns(max)

	// First saturate the connection pool.
	// Then start new requests for a connection that is canceled after it is requested.

	state = 1
	for i := 0; i < max; i++ {
		go func() {
			rows, err := db.Query("SELECT|people|name,photo|")
			if err != nil {
				t.Errorf("Query: %v", err)
				return
			}
			rows.Close()
			saturateDone.Done()
		}()
	}

	saturate.Wait()
	if t.Failed() {
		t.FailNow()
	}
	state = 2

	// Now cancel the request while it is waiting.
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	for i := 0; i < max; i++ {
		ctxReq, cancelReq := context.WithCancel(ctx)
		go func() {
			time.Sleep(100 * time.Millisecond)
			cancelReq()
		}()
		err := db.PingContext(ctxReq)
		if err != context.Canceled {
			t.Fatalf("PingContext (Exhaust): %v", err)
		}
	}
	close(donePing)
	saturateDone.Wait()

	// Now try to open a normal connection.
	err := db.PingContext(ctx)
	if err != nil {
		t.Fatalf("PingContext (Normal): %v", err)
	}
}

func TestRowsColumns(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	rows, err := db.Query("SELECT|people|age,name|")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	cols, err := rows.Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}
	want := []string{"age", "name"}
	if !slices.Equal(cols, want) {
		t.Errorf("got %#v; want %#v", cols, want)
	}
	if err := rows.Close(); err != nil {
		t.Errorf("error closing rows: %s", err)
	}
}

func TestRowsColumnTypes(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	rows, err := db.Query("SELECT|people|age,name|")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	tt, err := rows.ColumnTypes()
	if err != nil {
		t.Fatalf("ColumnTypes: %v", err)
	}

	types := make([]reflect.Type, len(tt))
	for i, tp := range tt {
		st := tp.ScanType()
		if st == nil {
			t.Errorf("scantype is null for column %q", tp.Name())
			continue
		}
		types[i] = st
	}
	values := make([]any, len(tt))
	for i := range values {
		values[i] = reflect.New(types[i]).Interface()
	}
	ct := 0
	for rows.Next() {
		err = rows.Scan(values...)
		if err != nil {
			t.Fatalf("failed to scan values in %v", err)
		}
		if ct == 1 {
			if age := *values[0].(*int32); age != 2 {
				t.Errorf("Expected 2, got %v", age)
			}
			if name := *values[1].(*string); name != "Bob" {
				t.Errorf("Expected Bob, got %v", name)
			}
		}
		ct++
	}
	if ct != 3 {
		t.Errorf("expected 3 rows, got %d", ct)
	}

	if err := rows.Close(); err != nil {
		t.Errorf("error closing rows: %s", err)
	}
}

func TestQueryRow(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	var name string
	var age int
	var birthday time.Time

	err := db.QueryRow("SELECT|people|age,name|age=?", 3).Scan(&age)
	if err == nil || !strings.Contains(err.Error(), "expected 2 destination arguments") {
		t.Errorf("expected error from wrong number of arguments; actually got: %v", err)
	}

	err = db.QueryRow("SELECT|people|bdate|age=?", 3).Scan(&birthday)
	if err != nil || !birthday.Equal(chrisBirthday) {
		t.Errorf("chris birthday = %v, err = %v; want %v", birthday, err, chrisBirthday)
	}

	err = db.QueryRow("SELECT|people|age,name|age=?", 2).Scan(&age, &name)
	if err != nil {
		t.Fatalf("age QueryRow+Scan: %v", err)
	}
	if name != "Bob" {
		t.Errorf("expected name Bob, got %q", name)
	}
	if age != 2 {
		t.Errorf("expected age 2, got %d", age)
	}

	err = db.QueryRow("SELECT|people|age,name|name=?", "Alice").Scan(&age, &name)
	if err != nil {
		t.Fatalf("name QueryRow+Scan: %v", err)
	}
	if name != "Alice" {
		t.Errorf("expected name Alice, got %q", name)
	}
	if age != 1 {
		t.Errorf("expected age 1, got %d", age)
	}

	var photo []byte
	err = db.QueryRow("SELECT|people|photo|name=?", "Alice").Scan(&photo)
	if err != nil {
		t.Fatalf("photo QueryRow+Scan: %v", err)
	}
	want := []byte("APHOTO")
	if !slices.Equal(photo, want) {
		t.Errorf("photo = %q; want %q", photo, want)
	}
}

func TestRowErr(t *testing.T) {
	db := newTestDB(t, "people")

	err := db.QueryRowContext(context.Background(), "SELECT|people|bdate|age=?", 3).Err()
	if err != nil {
		t.Errorf("Unexpected err = %v; want %v", err, nil)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err = db.QueryRowContext(ctx, "SELECT|people|bdate|age=?", 3).Err()
	exp := "context canceled"
	if err == nil || !strings.Contains(err.Error(), exp) {
		t.Errorf("Expected err = %v; got %v", exp, err)
	}
}

func TestTxRollbackCommitErr(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	err = tx.Rollback()
	if err != nil {
		t.Errorf("expected nil error from Rollback; got %v", err)
	}
	err = tx.Commit()
	if err != ErrTxDone {
		t.Errorf("expected %q from Commit; got %q", ErrTxDone, err)
	}

	tx, err = db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	err = tx.Commit()
	if err != nil {
		t.Errorf("expected nil error from Commit; got %v", err)
	}
	err = tx.Rollback()
	if err != ErrTxDone {
		t.Errorf("expected %q from Rollback; got %q", ErrTxDone, err)
	}
}

func TestStatementErrorAfterClose(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	stmt, err := db.Prepare("SELECT|people|age|name=?")
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	err = stmt.Close()
	if err != nil {
		t.Fatalf("Close: %v", err)
	}
	var name string
	err = stmt.QueryRow("foo").Scan(&name)
	if err == nil {
		t.Errorf("expected error from QueryRow.Scan after Stmt.Close")
	}
}

func TestStatementQueryRow(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	stmt, err := db.Prepare("SELECT|people|age|name=?")
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	defer stmt.Close()
	var age int
	for n, tt := range []struct {
		name string
		want int
	}{
		{"Alice", 1},
		{"Bob", 2},
		{"Chris", 3},
	} {
		if err := stmt.QueryRow(tt.name).Scan(&age); err != nil {
			t.Errorf("%d: on %q, QueryRow/Scan: %v", n, tt.name, err)
		} else if age != tt.want {
			t.Errorf("%d: age=%d, want %d", n, age, tt.want)
		}
	}
}

type stubDriverStmt struct {
	err error
}

func (s stubDriverStmt) Close() error {
	return s.err
}

func (s stubDriverStmt) NumInput() int {
	return -1
}

func (s stubDriverStmt) Exec(args []driver.Value) (driver.Result, error) {
	return nil, nil
}

func (s stubDriverStmt) Query(args []driver.Value) (driver.Rows, error) {
	return nil, nil
}

// golang.org/issue/12798
func TestStatementClose(t *testing.T) {
	want := errors.New("STMT ERROR")

	tests := []struct {
		stmt *Stmt
		msg  string
	}{
		{&Stmt{stickyErr: want}, "stickyErr not propagated"},
		{&Stmt{cg: &Tx{}, cgds: &driverStmt{Locker: &sync.Mutex{}, si: stubDriverStmt{want}}}, "driverStmt.Close() error not propagated"},
	}
	for _, test := range tests {
		if err := test.stmt.Close(); err != want {
			t.Errorf("%s. Got stmt.Close() = %v, want = %v", test.msg, err, want)
		}
	}
}

// golang.org/issue/3734
func TestStatementQueryRowConcurrent(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	stmt, err := db.Prepare("SELECT|people|age|name=?")
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	defer stmt.Close()

	const n = 10
	ch := make(chan error, n)
	for i := 0; i < n; i++ {
		go func() {
			var age int
			err := stmt.QueryRow("Alice").Scan(&age)
			if err == nil && age != 1 {
				err = fmt.Errorf("unexpected age %d", age)
			}
			ch <- err
		}()
	}
	for i := 0; i < n; i++ {
		if err := <-ch; err != nil {
			t.Error(err)
		}
	}
}

// just a test of fakedb itself
func TestBogusPreboundParameters(t *testing.T) {
	db := newTestDB(t, "foo")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32,dead=bool")
	_, err := db.Prepare("INSERT|t1|name=?,age=bogusconversion")
	if err == nil {
		t.Fatalf("expected error")
	}
	if err.Error() != `fakedb: invalid conversion to int32 from "bogusconversion"` {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestExec(t *testing.T) {
	db := newTestDB(t, "foo")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32,dead=bool")
	stmt, err := db.Prepare("INSERT|t1|name=?,age=?")
	if err != nil {
		t.Errorf("Stmt, err = %v, %v", stmt, err)
	}
	defer stmt.Close()

	type execTest struct {
		args    []any
		wantErr string
	}
	execTests := []execTest{
		// Okay:
		{[]any{"Brad", 31}, ""},
		{[]any{"Brad", int64(31)}, ""},
		{[]any{"Bob", "32"}, ""},
		{[]any{7, 9}, ""},

		// Invalid conversions:
		{[]any{"Brad", int64(0xFFFFFFFF)}, "sql: converting argument $2 type: sql/driver: value 4294967295 overflows int32"},
		{[]any{"Brad", "strconv fail"}, `sql: converting argument $2 type: sql/driver: value "strconv fail" can't be converted to int32`},

		// Wrong number of args:
		{[]any{}, "sql: expected 2 arguments, got 0"},
		{[]any{1, 2, 3}, "sql: expected 2 arguments, got 3"},
	}
	for n, et := range execTests {
		_, err := stmt.Exec(et.args...)
		errStr := ""
		if err != nil {
			errStr = err.Error()
		}
		if errStr != et.wantErr {
			t.Errorf("stmt.Execute #%d: for %v, got error %q, want error %q",
				n, et.args, errStr, et.wantErr)
		}
	}
}

func TestTxPrepare(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32,dead=bool")
	tx, err := db.Begin()
	if err != nil {
		t.Fatalf("Begin = %v", err)
	}
	stmt, err := tx.Prepare("INSERT|t1|name=?,age=?")
	if err != nil {
		t.Fatalf("Stmt, err = %v, %v", stmt, err)
	}
	defer stmt.Close()
	_, err = stmt.Exec("Bobby", 7)
	if err != nil {
		t.Fatalf("Exec = %v", err)
	}
	err = tx.Commit()
	if err != nil {
		t.Fatalf("Commit = %v", err)
	}
	// Commit() should have closed the statement
	if !stmt.closed {
		t.Fatal("Stmt not closed after Commit")
	}
}

func TestTxStmt(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32,dead=bool")
	stmt, err := db.Prepare("INSERT|t1|name=?,age=?")
	if err != nil {
		t.Fatalf("Stmt, err = %v, %v", stmt, err)
	}
	defer stmt.Close()
	tx, err := db.Begin()
	if err != nil {
		t.Fatalf("Begin = %v", err)
	}
	txs := tx.Stmt(stmt)
	defer txs.Close()
	_, err = txs.Exec("Bobby", 7)
	if err != nil {
		t.Fatalf("Exec = %v", err)
	}
	err = tx.Commit()
	if err != nil {
		t.Fatalf("Commit = %v", err)
	}
	// Commit() should have closed the statement
	if !txs.closed {
		t.Fatal("Stmt not closed after Commit")
	}
}

func TestTxStmtPreparedOnce(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32")

	prepares0 := numPrepares(t, db)

	// db.Prepare increments numPrepares.
	stmt, err := db.Prepare("INSERT|t1|name=?,age=?")
	if err != nil {
		t.Fatalf("Stmt, err = %v, %v", stmt, err)
	}
	defer stmt.Close()

	tx, err := db.Begin()
	if err != nil {
		t.Fatalf("Begin = %v", err)
	}

	txs1 := tx.Stmt(stmt)
	txs2 := tx.Stmt(stmt)

	_, err = txs1.Exec("Go", 7)
	if err != nil {
		t.Fatalf("Exec = %v", err)
	}
	txs1.Close()

	_, err = txs2.Exec("Gopher", 8)
	if err != nil {
		t.Fatalf("Exec = %v", err)
	}
	txs2.Close()

	err = tx.Commit()
	if err != nil {
		t.Fatalf("Commit = %v", err)
	}

	if prepares := numPrepares(t, db) - prepares0; prepares != 1 {
		t.Errorf("executed %d Prepare statements; want 1", prepares)
	}
}

func TestTxStmtClosedRePrepares(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32")

	prepares0 := numPrepares(t, db)

	// db.Prepare increments numPrepares.
	stmt, err := db.Prepare("INSERT|t1|name=?,age=?")
	if err != nil {
		t.Fatalf("Stmt, err = %v, %v", stmt, err)
	}
	tx, err := db.Begin()
	if err != nil {
		t.Fatalf("Begin = %v", err)
	}
	err = stmt.Close()
	if err != nil {
		t.Fatalf("stmt.Close() = %v", err)
	}
	// tx.Stmt increments numPrepares because stmt is closed.
	txs := tx.Stmt(stmt)
	if txs.stickyErr != nil {
		t.Fatal(txs.stickyErr)
	}
	if txs.parentStmt != nil {
		t.Fatal("expected nil parentStmt")
	}
	_, err = txs.Exec(`Eric`, 82)
	if err != nil {
		t.Fatalf("txs.Exec = %v", err)
	}

	err = txs.Close()
	if err != nil {
		t.Fatalf("txs.Close = %v", err)
	}

	tx.Rollback()

	if prepares := numPrepares(t, db) - prepares0; prepares != 2 {
		t.Errorf("executed %d Prepare statements; want 2", prepares)
	}
}

func TestParentStmtOutlivesTxStmt(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32")

	// Make sure everything happens on the same connection.
	db.SetMaxOpenConns(1)

	prepares0 := numPrepares(t, db)

	// db.Prepare increments numPrepares.
	stmt, err := db.Prepare("INSERT|t1|name=?,age=?")
	if err != nil {
		t.Fatalf("Stmt, err = %v, %v", stmt, err)
	}
	defer stmt.Close()
	tx, err := db.Begin()
	if err != nil {
		t.Fatalf("Begin = %v", err)
	}
	txs := tx.Stmt(stmt)
	if len(stmt.css) != 1 {
		t.Fatalf("len(stmt.css) = %v; want 1", len(stmt.css))
	}
	err = txs.Close()
	if err != nil {
		t.Fatalf("txs.Close() = %v", err)
	}
	err = tx.Rollback()
	if err != nil {
		t.Fatalf("tx.Rollback() = %v", err)
	}
	// txs must not be valid.
	_, err = txs.Exec("Suzan", 30)
	if err == nil {
		t.Fatalf("txs.Exec(), expected err")
	}
	// Stmt must still be valid.
	_, err = stmt.Exec("Janina", 25)
	if err != nil {
		t.Fatalf("stmt.Exec() = %v", err)
	}

	if prepares := numPrepares(t, db) - prepares0; prepares != 1 {
		t.Errorf("executed %d Prepare statements; want 1", prepares)
	}
}

// Test that tx.Stmt called with a statement already
// associated with tx as argument re-prepares the same
// statement again.
func TestTxStmtFromTxStmtRePrepares(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32")
	prepares0 := numPrepares(t, db)
	// db.Prepare increments numPrepares.
	stmt, err := db.Prepare("INSERT|t1|name=?,age=?")
	if err != nil {
		t.Fatalf("Stmt, err = %v, %v", stmt, err)
	}
	defer stmt.Close()

	tx, err := db.Begin()
	if err != nil {
		t.Fatalf("Begin = %v", err)
	}
	txs1 := tx.Stmt(stmt)

	// tx.Stmt(txs1) increments numPrepares because txs1 already
	// belongs to a transaction (albeit the same transaction).
	txs2 := tx.Stmt(txs1)
	if txs2.stickyErr != nil {
		t.Fatal(txs2.stickyErr)
	}
	if txs2.parentStmt != nil {
		t.Fatal("expected nil parentStmt")
	}
	_, err = txs2.Exec(`Eric`, 82)
	if err != nil {
		t.Fatal(err)
	}

	err = txs1.Close()
	if err != nil {
		t.Fatalf("txs1.Close = %v", err)
	}
	err = txs2.Close()
	if err != nil {
		t.Fatalf("txs1.Close = %v", err)
	}
	err = tx.Rollback()
	if err != nil {
		t.Fatalf("tx.Rollback = %v", err)
	}

	if prepares := numPrepares(t, db) - prepares0; prepares != 2 {
		t.Errorf("executed %d Prepare statements; want 2", prepares)
	}
}

// Issue: https://golang.org/issue/2784
// This test didn't fail before because we got lucky with the fakedb driver.
// It was failing, and now not, in github.com/bradfitz/go-sql-test
func TestTxQuery(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32,dead=bool")
	exec(t, db, "INSERT|t1|name=Alice")

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback()

	r, err := tx.Query("SELECT|t1|name|")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if !r.Next() {
		if r.Err() != nil {
			t.Fatal(r.Err())
		}
		t.Fatal("expected one row")
	}

	var x string
	err = r.Scan(&x)
	if err != nil {
		t.Fatal(err)
	}
}

func TestTxQueryInvalid(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback()

	_, err = tx.Query("SELECT|t1|name|")
	if err == nil {
		t.Fatal("Error expected")
	}
}

// Tests fix for issue 4433, that retries in Begin happen when
// conn.Begin() returns ErrBadConn
func TestTxErrBadConn(t *testing.T) {
	db, err := Open("test", fakeDBName+";badConn")
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := db.Exec("WIPE"); err != nil {
		t.Fatalf("exec wipe: %v", err)
	}
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32,dead=bool")
	stmt, err := db.Prepare("INSERT|t1|name=?,age=?")
	if err != nil {
		t.Fatalf("Stmt, err = %v, %v", stmt, err)
	}
	defer stmt.Close()
	tx, err := db.Begin()
	if err != nil {
		t.Fatalf("Begin = %v", err)
	}
	txs := tx.Stmt(stmt)
	defer txs.Close()
	_, err = txs.Exec("Bobby", 7)
	if err != nil {
		t.Fatalf("Exec = %v", err)
	}
	err = tx.Commit()
	if err != nil {
		t.Fatalf("Commit = %v", err)
	}
}

func TestConnQuery(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	conn, err := db.Conn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	conn.dc.ci.(*fakeConn).skipDirtySession = true
	defer conn.Close()

	var name string
	err = conn.QueryRowContext(ctx, "SELECT|people|name|age=?", 3).Scan(&name)
	if err != nil {
		t.Fatal(err)
	}
	if name != "Chris" {
		t.Fatalf("unexpected result, got %q want Chris", name)
	}

	err = conn.PingContext(ctx)
	if err != nil {
		t.Fatal(err)
	}
}

func TestConnRaw(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	conn, err := db.Conn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	conn.dc.ci.(*fakeConn).skipDirtySession = true
	defer conn.Close()

	sawFunc := false
	err = conn.Raw(func { dc ->
		sawFunc = true
		if _, ok := dc.(*fakeConn); !ok {
			return fmt.Errorf("got %T want *fakeConn", dc)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !sawFunc {
		t.Fatal("Raw func not called")
	}

	func() {
		defer func() {
			x := recover()
			if x == nil {
				t.Fatal("expected panic")
			}
			conn.closemu.Lock()
			closed := conn.dc == nil
			conn.closemu.Unlock()
			if !closed {
				t.Fatal("expected connection to be closed after panic")
			}
		}()
		err = conn.Raw(func { dc -> panic("Conn.Raw panic should return an error") })
		t.Fatal("expected panic from Raw func")
	}()
}

func TestCursorFake(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	exec(t, db, "CREATE|peoplecursor|list=table")
	exec(t, db, "INSERT|peoplecursor|list=people!name!age")

	rows, err := db.QueryContext(ctx, `SELECT|peoplecursor|list|`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()

	if !rows.Next() {
		t.Fatal("no rows")
	}
	var cursor = &Rows{}
	err = rows.Scan(cursor)
	if err != nil {
		t.Fatal(err)
	}
	defer cursor.Close()

	const expectedRows = 3
	var currentRow int64

	var n int64
	var s string
	for cursor.Next() {
		currentRow++
		err = cursor.Scan(&s, &n)
		if err != nil {
			t.Fatal(err)
		}
		if n != currentRow {
			t.Errorf("expected number(Age)=%d, got %d", currentRow, n)
		}
	}
	if currentRow != expectedRows {
		t.Errorf("expected %d rows, got %d rows", expectedRows, currentRow)
	}
}

func TestInvalidNilValues(t *testing.T) {
	var date1 time.Time
	var date2 int

	tests := []struct {
		name          string
		input         any
		expectedError string
	}{
		{
			name:          "time.Time",
			input:         &date1,
			expectedError: `sql: Scan error on column index 0, name "bdate": unsupported Scan, storing driver.Value type <nil> into type *time.Time`,
		},
		{
			name:          "int",
			input:         &date2,
			expectedError: `sql: Scan error on column index 0, name "bdate": converting NULL to int is unsupported`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func { t ->
			db := newTestDB(t, "people")
			defer closeDB(t, db)

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			conn, err := db.Conn(ctx)
			if err != nil {
				t.Fatal(err)
			}
			conn.dc.ci.(*fakeConn).skipDirtySession = true
			defer conn.Close()

			err = conn.QueryRowContext(ctx, "SELECT|people|bdate|age=?", 1).Scan(tt.input)
			if err == nil {
				t.Fatal("expected error when querying nil column, but succeeded")
			}
			if err.Error() != tt.expectedError {
				t.Fatalf("Expected error: %s\nReceived: %s", tt.expectedError, err.Error())
			}

			err = conn.PingContext(ctx)
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestConnTx(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	conn, err := db.Conn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	conn.dc.ci.(*fakeConn).skipDirtySession = true
	defer conn.Close()

	tx, err := conn.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	insertName, insertAge := "Nancy", 33
	_, err = tx.ExecContext(ctx, "INSERT|people|name=?,age=?,photo=APHOTO", insertName, insertAge)
	if err != nil {
		t.Fatal(err)
	}
	err = tx.Commit()
	if err != nil {
		t.Fatal(err)
	}

	var selectName string
	err = conn.QueryRowContext(ctx, "SELECT|people|name|age=?", insertAge).Scan(&selectName)
	if err != nil {
		t.Fatal(err)
	}
	if selectName != insertName {
		t.Fatalf("got %q want %q", selectName, insertName)
	}
}

// TestConnIsValid verifies that a database connection that should be discarded,
// is actually discarded and does not re-enter the connection pool.
// If the IsValid method from *fakeConn is removed, this test will fail.
func TestConnIsValid(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	db.SetMaxOpenConns(1)

	ctx := context.Background()

	c, err := db.Conn(ctx)
	if err != nil {
		t.Fatal(err)
	}

	err = c.Raw(func { raw ->
		dc := raw.(*fakeConn)
		dc.stickyBad = true
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	c.Close()

	if len(db.freeConn) > 0 && db.freeConn[0].ci.(*fakeConn).stickyBad {
		t.Fatal("bad connection returned to pool; expected bad connection to be discarded")
	}
}

// Tests fix for issue 2542, that we release a lock when querying on
// a closed connection.
func TestIssue2542Deadlock(t *testing.T) {
	db := newTestDB(t, "people")
	closeDB(t, db)
	for i := 0; i < 2; i++ {
		_, err := db.Query("SELECT|people|age,name|")
		if err == nil {
			t.Fatalf("expected error")
		}
	}
}

// From golang.org/issue/3865
func TestCloseStmtBeforeRows(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	s, err := db.Prepare("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}

	r, err := s.Query()
	if err != nil {
		s.Close()
		t.Fatal(err)
	}

	err = s.Close()
	if err != nil {
		t.Fatal(err)
	}

	r.Close()
}

// Tests fix for issue 2788, that we bind nil to a []byte if the
// value in the column is sql null
func TestNullByteSlice(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t|id=int32,name=nullstring")
	exec(t, db, "INSERT|t|id=10,name=?", nil)

	var name []byte

	err := db.QueryRow("SELECT|t|name|id=?", 10).Scan(&name)
	if err != nil {
		t.Fatal(err)
	}
	if name != nil {
		t.Fatalf("name []byte should be nil for null column value, got: %#v", name)
	}

	exec(t, db, "INSERT|t|id=11,name=?", "bob")
	err = db.QueryRow("SELECT|t|name|id=?", 11).Scan(&name)
	if err != nil {
		t.Fatal(err)
	}
	if string(name) != "bob" {
		t.Fatalf("name []byte should be bob, got: %q", string(name))
	}
}

func TestPointerParamsAndScans(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t|id=int32,name=nullstring")

	bob := "bob"
	var name *string

	name = &bob
	exec(t, db, "INSERT|t|id=10,name=?", name)
	name = nil
	exec(t, db, "INSERT|t|id=20,name=?", name)

	err := db.QueryRow("SELECT|t|name|id=?", 10).Scan(&name)
	if err != nil {
		t.Fatalf("querying id 10: %v", err)
	}
	if name == nil {
		t.Errorf("id 10's name = nil; want bob")
	} else if *name != "bob" {
		t.Errorf("id 10's name = %q; want bob", *name)
	}

	err = db.QueryRow("SELECT|t|name|id=?", 20).Scan(&name)
	if err != nil {
		t.Fatalf("querying id 20: %v", err)
	}
	if name != nil {
		t.Errorf("id 20 = %q; want nil", *name)
	}
}

func TestQueryRowClosingStmt(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	var name string
	var age int
	err := db.QueryRow("SELECT|people|age,name|age=?", 3).Scan(&age, &name)
	if err != nil {
		t.Fatal(err)
	}
	if len(db.freeConn) != 1 {
		t.Fatalf("expected 1 free conn")
	}
	fakeConn := db.freeConn[0].ci.(*fakeConn)
	if made, closed := fakeConn.stmtsMade, fakeConn.stmtsClosed; made != closed {
		t.Errorf("statement close mismatch: made %d, closed %d", made, closed)
	}
}

var atomicRowsCloseHook atomic.Value // of func(*Rows, *error)

func init() {
	rowsCloseHook = func {
		fn, _ := atomicRowsCloseHook.Load().(func(*Rows, *error))
		return fn
	}
}

func setRowsCloseHook(fn func(*Rows, *error)) {
	if fn == nil {
		// Can't change an atomic.Value back to nil, so set it to this
		// no-op func instead.
		fn = func {}
	}
	atomicRowsCloseHook.Store(fn)
}

// Test issue 6651
func TestIssue6651(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	var v string

	want := "error in rows.Next"
	rowsCursorNextHook = func { dest -> errors.New(want) }
	defer func() { rowsCursorNextHook = nil }()

	err := db.QueryRow("SELECT|people|name|").Scan(&v)
	if err == nil || err.Error() != want {
		t.Errorf("error = %q; want %q", err, want)
	}
	rowsCursorNextHook = nil

	want = "error in rows.Close"
	setRowsCloseHook(func { rows, err -> *err = errors.New(want) })
	defer setRowsCloseHook(nil)
	err = db.QueryRow("SELECT|people|name|").Scan(&v)
	if err == nil || err.Error() != want {
		t.Errorf("error = %q; want %q", err, want)
	}
}

type nullTestRow struct {
	nullParam    any
	notNullParam any
	scanNullVal  any
}

type nullTestSpec struct {
	nullType    string
	notNullType string
	rows        [6]nullTestRow
}

func TestNullStringParam(t *testing.T) {
	spec := nullTestSpec{"nullstring", "string", [6]nullTestRow{
		{NullString{"aqua", true}, "", NullString{"aqua", true}},
		{NullString{"brown", false}, "", NullString{"", false}},
		{"chartreuse", "", NullString{"chartreuse", true}},
		{NullString{"darkred", true}, "", NullString{"darkred", true}},
		{NullString{"eel", false}, "", NullString{"", false}},
		{"foo", NullString{"black", false}, nil},
	}}
	nullTestRun(t, spec)
}

func TestGenericNullStringParam(t *testing.T) {
	spec := nullTestSpec{"nullstring", "string", [6]nullTestRow{
		{Null[string]{"aqua", true}, "", Null[string]{"aqua", true}},
		{Null[string]{"brown", false}, "", Null[string]{"", false}},
		{"chartreuse", "", Null[string]{"chartreuse", true}},
		{Null[string]{"darkred", true}, "", Null[string]{"darkred", true}},
		{Null[string]{"eel", false}, "", Null[string]{"", false}},
		{"foo", Null[string]{"black", false}, nil},
	}}
	nullTestRun(t, spec)
}

func TestNullInt64Param(t *testing.T) {
	spec := nullTestSpec{"nullint64", "int64", [6]nullTestRow{
		{NullInt64{31, true}, 1, NullInt64{31, true}},
		{NullInt64{-22, false}, 1, NullInt64{0, false}},
		{22, 1, NullInt64{22, true}},
		{NullInt64{33, true}, 1, NullInt64{33, true}},
		{NullInt64{222, false}, 1, NullInt64{0, false}},
		{0, NullInt64{31, false}, nil},
	}}
	nullTestRun(t, spec)
}

func TestNullInt32Param(t *testing.T) {
	spec := nullTestSpec{"nullint32", "int32", [6]nullTestRow{
		{NullInt32{31, true}, 1, NullInt32{31, true}},
		{NullInt32{-22, false}, 1, NullInt32{0, false}},
		{22, 1, NullInt32{22, true}},
		{NullInt32{33, true}, 1, NullInt32{33, true}},
		{NullInt32{222, false}, 1, NullInt32{0, false}},
		{0, NullInt32{31, false}, nil},
	}}
	nullTestRun(t, spec)
}

func TestNullInt16Param(t *testing.T) {
	spec := nullTestSpec{"nullint16", "int16", [6]nullTestRow{
		{NullInt16{31, true}, 1, NullInt16{31, true}},
		{NullInt16{-22, false}, 1, NullInt16{0, false}},
		{22, 1, NullInt16{22, true}},
		{NullInt16{33, true}, 1, NullInt16{33, true}},
		{NullInt16{222, false}, 1, NullInt16{0, false}},
		{0, NullInt16{31, false}, nil},
	}}
	nullTestRun(t, spec)
}

func TestNullByteParam(t *testing.T) {
	spec := nullTestSpec{"nullbyte", "byte", [6]nullTestRow{
		{NullByte{31, true}, 1, NullByte{31, true}},
		{NullByte{0, false}, 1, NullByte{0, false}},
		{22, 1, NullByte{22, true}},
		{NullByte{33, true}, 1, NullByte{33, true}},
		{NullByte{222, false}, 1, NullByte{0, false}},
		{0, NullByte{31, false}, nil},
	}}
	nullTestRun(t, spec)
}

func TestNullFloat64Param(t *testing.T) {
	spec := nullTestSpec{"nullfloat64", "float64", [6]nullTestRow{
		{NullFloat64{31.2, true}, 1, NullFloat64{31.2, true}},
		{NullFloat64{13.1, false}, 1, NullFloat64{0, false}},
		{-22.9, 1, NullFloat64{-22.9, true}},
		{NullFloat64{33.81, true}, 1, NullFloat64{33.81, true}},
		{NullFloat64{222, false}, 1, NullFloat64{0, false}},
		{10, NullFloat64{31.2, false}, nil},
	}}
	nullTestRun(t, spec)
}

func TestNullBoolParam(t *testing.T) {
	spec := nullTestSpec{"nullbool", "bool", [6]nullTestRow{
		{NullBool{false, true}, true, NullBool{false, true}},
		{NullBool{true, false}, false, NullBool{false, false}},
		{true, true, NullBool{true, true}},
		{NullBool{true, true}, false, NullBool{true, true}},
		{NullBool{true, false}, true, NullBool{false, false}},
		{true, NullBool{true, false}, nil},
	}}
	nullTestRun(t, spec)
}

func TestNullTimeParam(t *testing.T) {
	t0 := time.Time{}
	t1 := time.Date(2000, 1, 1, 8, 9, 10, 11, time.UTC)
	t2 := time.Date(2010, 1, 1, 8, 9, 10, 11, time.UTC)
	spec := nullTestSpec{"nulldatetime", "datetime", [6]nullTestRow{
		{NullTime{t1, true}, t2, NullTime{t1, true}},
		{NullTime{t1, false}, t2, NullTime{t0, false}},
		{t1, t2, NullTime{t1, true}},
		{NullTime{t1, true}, t2, NullTime{t1, true}},
		{NullTime{t1, false}, t2, NullTime{t0, false}},
		{t2, NullTime{t1, false}, nil},
	}}
	nullTestRun(t, spec)
}

func nullTestRun(t *testing.T, spec nullTestSpec) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, fmt.Sprintf("CREATE|t|id=int32,name=string,nullf=%s,notnullf=%s", spec.nullType, spec.notNullType))

	// Inserts with db.Exec:
	exec(t, db, "INSERT|t|id=?,name=?,nullf=?,notnullf=?", 1, "alice", spec.rows[0].nullParam, spec.rows[0].notNullParam)
	exec(t, db, "INSERT|t|id=?,name=?,nullf=?,notnullf=?", 2, "bob", spec.rows[1].nullParam, spec.rows[1].notNullParam)

	// Inserts with a prepared statement:
	stmt, err := db.Prepare("INSERT|t|id=?,name=?,nullf=?,notnullf=?")
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	defer stmt.Close()
	if _, err := stmt.Exec(3, "chris", spec.rows[2].nullParam, spec.rows[2].notNullParam); err != nil {
		t.Errorf("exec insert chris: %v", err)
	}
	if _, err := stmt.Exec(4, "dave", spec.rows[3].nullParam, spec.rows[3].notNullParam); err != nil {
		t.Errorf("exec insert dave: %v", err)
	}
	if _, err := stmt.Exec(5, "eleanor", spec.rows[4].nullParam, spec.rows[4].notNullParam); err != nil {
		t.Errorf("exec insert eleanor: %v", err)
	}

	// Can't put null val into non-null col
	row5 := spec.rows[5]
	if _, err := stmt.Exec(6, "bob", row5.nullParam, row5.notNullParam); err == nil {
		t.Errorf("expected error inserting nil val with prepared statement Exec: NULL=%#v, NOT-NULL=%#v", row5.nullParam, row5.notNullParam)
	}

	_, err = db.Exec("INSERT|t|id=?,name=?,nullf=?", 999, nil, nil)
	if err == nil {
		// TODO: this test fails, but it's just because
		// fakeConn implements the optional Execer interface,
		// so arguably this is the correct behavior. But
		// maybe I should flesh out the fakeConn.Exec
		// implementation so this properly fails.
		// t.Errorf("expected error inserting nil name with Exec")
	}

	paramtype := reflect.TypeOf(spec.rows[0].nullParam)
	bindVal := reflect.New(paramtype).Interface()

	for i := 0; i < 5; i++ {
		id := i + 1
		if err := db.QueryRow("SELECT|t|nullf|id=?", id).Scan(bindVal); err != nil {
			t.Errorf("id=%d Scan: %v", id, err)
		}
		bindValDeref := reflect.ValueOf(bindVal).Elem().Interface()
		if !reflect.DeepEqual(bindValDeref, spec.rows[i].scanNullVal) {
			t.Errorf("id=%d got %#v, want %#v", id, bindValDeref, spec.rows[i].scanNullVal)
		}
	}
}

// golang.org/issue/4859
func TestQueryRowNilScanDest(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	var name *string // nil pointer
	err := db.QueryRow("SELECT|people|name|").Scan(name)
	want := `sql: Scan error on column index 0, name "name": destination pointer is nil`
	if err == nil || err.Error() != want {
		t.Errorf("error = %q; want %q", err.Error(), want)
	}
}

func TestIssue4902(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	driver := db.Driver().(*fakeDriver)
	opens0 := driver.openCount

	var stmt *Stmt
	var err error
	for i := 0; i < 10; i++ {
		stmt, err = db.Prepare("SELECT|people|name|")
		if err != nil {
			t.Fatal(err)
		}
		err = stmt.Close()
		if err != nil {
			t.Fatal(err)
		}
	}

	opens := driver.openCount - opens0
	if opens > 1 {
		t.Errorf("opens = %d; want <= 1", opens)
		t.Logf("db = %#v", db)
		t.Logf("driver = %#v", driver)
		t.Logf("stmt = %#v", stmt)
	}
}

// Issue 3857
// This used to deadlock.
func TestSimultaneousQueries(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback()

	r1, err := tx.Query("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	defer r1.Close()

	r2, err := tx.Query("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	defer r2.Close()
}

func TestMaxIdleConns(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	tx.Commit()
	if got := len(db.freeConn); got != 1 {
		t.Errorf("freeConns = %d; want 1", got)
	}

	db.SetMaxIdleConns(0)

	if got := len(db.freeConn); got != 0 {
		t.Errorf("freeConns after set to zero = %d; want 0", got)
	}

	tx, err = db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	tx.Commit()
	if got := len(db.freeConn); got != 0 {
		t.Errorf("freeConns = %d; want 0", got)
	}
}

func TestMaxOpenConns(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	defer setHookpostCloseConn(nil)
	setHookpostCloseConn(func { _, err -> if err != nil {
		t.Errorf("Error closing fakeConn: %v", err)
	} })

	db := newTestDB(t, "magicquery")
	defer closeDB(t, db)

	driver := db.Driver().(*fakeDriver)

	// Force the number of open connections to 0 so we can get an accurate
	// count for the test
	db.clearAllConns(t)

	driver.mu.Lock()
	opens0 := driver.openCount
	closes0 := driver.closeCount
	driver.mu.Unlock()

	db.SetMaxIdleConns(10)
	db.SetMaxOpenConns(10)

	stmt, err := db.Prepare("SELECT|magicquery|op|op=?,millis=?")
	if err != nil {
		t.Fatal(err)
	}

	// Start 50 parallel slow queries.
	const (
		nquery      = 50
		sleepMillis = 25
		nbatch      = 2
	)
	var wg sync.WaitGroup
	for batch := 0; batch < nbatch; batch++ {
		for i := 0; i < nquery; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				var op string
				if err := stmt.QueryRow("sleep", sleepMillis).Scan(&op); err != nil && err != ErrNoRows {
					t.Error(err)
				}
			}()
		}
		// Wait for the batch of queries above to finish before starting the next round.
		wg.Wait()
	}

	if g, w := db.numFreeConns(), 10; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	if n := db.numDepsPoll(t, 20); n > 20 {
		t.Errorf("number of dependencies = %d; expected <= 20", n)
		db.dumpDeps(t)
	}

	driver.mu.Lock()
	opens := driver.openCount - opens0
	closes := driver.closeCount - closes0
	driver.mu.Unlock()

	if opens > 10 {
		t.Logf("open calls = %d", opens)
		t.Logf("close calls = %d", closes)
		t.Errorf("db connections opened = %d; want <= 10", opens)
		db.dumpDeps(t)
	}

	if err := stmt.Close(); err != nil {
		t.Fatal(err)
	}

	if g, w := db.numFreeConns(), 10; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	if n := db.numDepsPoll(t, 10); n > 10 {
		t.Errorf("number of dependencies = %d; expected <= 10", n)
		db.dumpDeps(t)
	}

	db.SetMaxOpenConns(5)

	if g, w := db.numFreeConns(), 5; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	if n := db.numDepsPoll(t, 5); n > 5 {
		t.Errorf("number of dependencies = %d; expected 0", n)
		db.dumpDeps(t)
	}

	db.SetMaxOpenConns(0)

	if g, w := db.numFreeConns(), 5; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	if n := db.numDepsPoll(t, 5); n > 5 {
		t.Errorf("number of dependencies = %d; expected 0", n)
		db.dumpDeps(t)
	}

	db.clearAllConns(t)
}

// Issue 9453: tests that SetMaxOpenConns can be lowered at runtime
// and affects the subsequent release of connections.
func TestMaxOpenConnsOnBusy(t *testing.T) {
	defer setHookpostCloseConn(nil)
	setHookpostCloseConn(func { _, err -> if err != nil {
		t.Errorf("Error closing fakeConn: %v", err)
	} })

	db := newTestDB(t, "magicquery")
	defer closeDB(t, db)

	db.SetMaxOpenConns(3)

	ctx := context.Background()

	conn0, err := db.conn(ctx, cachedOrNewConn)
	if err != nil {
		t.Fatalf("db open conn fail: %v", err)
	}

	conn1, err := db.conn(ctx, cachedOrNewConn)
	if err != nil {
		t.Fatalf("db open conn fail: %v", err)
	}

	conn2, err := db.conn(ctx, cachedOrNewConn)
	if err != nil {
		t.Fatalf("db open conn fail: %v", err)
	}

	if g, w := db.numOpen, 3; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	db.SetMaxOpenConns(2)
	if g, w := db.numOpen, 3; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	conn0.releaseConn(nil)
	conn1.releaseConn(nil)
	if g, w := db.numOpen, 2; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	conn2.releaseConn(nil)
	if g, w := db.numOpen, 2; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}
}

// Issue 10886: tests that all connection attempts return when more than
// DB.maxOpen connections are in flight and the first DB.maxOpen fail.
func TestPendingConnsAfterErr(t *testing.T) {
	const (
		maxOpen = 2
		tryOpen = maxOpen*2 + 2
	)

	// No queries will be run.
	db, err := Open("test", fakeDBName)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer closeDB(t, db)
	defer func() {
		for k, v := range db.lastPut {
			t.Logf("%p: %v", k, v)
		}
	}()

	db.SetMaxOpenConns(maxOpen)
	db.SetMaxIdleConns(0)

	errOffline := errors.New("db offline")

	defer func() { setHookOpenErr(nil) }()

	errs := make(chan error, tryOpen)

	var opening sync.WaitGroup
	opening.Add(tryOpen)

	setHookOpenErr(func {
		// Wait for all connections to enqueue.
		opening.Wait()
		return errOffline
	})

	for i := 0; i < tryOpen; i++ {
		go func() {
			opening.Done() // signal one connection is in flight
			_, err := db.Exec("will never run")
			errs <- err
		}()
	}

	opening.Wait() // wait for all workers to begin running

	const timeout = 5 * time.Second
	to := time.NewTimer(timeout)
	defer to.Stop()

	// check that all connections fail without deadlock
	for i := 0; i < tryOpen; i++ {
		select {
		case err := <-errs:
			if got, want := err, errOffline; got != want {
				t.Errorf("unexpected err: got %v, want %v", got, want)
			}
		case <-to.C:
			t.Fatalf("orphaned connection request(s), still waiting after %v", timeout)
		}
	}

	// Wait a reasonable time for the database to close all connections.
	tick := time.NewTicker(3 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-tick.C:
			db.mu.Lock()
			if db.numOpen == 0 {
				db.mu.Unlock()
				return
			}
			db.mu.Unlock()
		case <-to.C:
			// Closing the database will check for numOpen and fail the test.
			return
		}
	}
}

func TestSingleOpenConn(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	db.SetMaxOpenConns(1)

	rows, err := db.Query("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	if err = rows.Close(); err != nil {
		t.Fatal(err)
	}
	// shouldn't deadlock
	rows, err = db.Query("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	if err = rows.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestStats(t *testing.T) {
	db := newTestDB(t, "people")
	stats := db.Stats()
	if got := stats.OpenConnections; got != 1 {
		t.Errorf("stats.OpenConnections = %d; want 1", got)
	}

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	tx.Commit()

	closeDB(t, db)
	stats = db.Stats()
	if got := stats.OpenConnections; got != 0 {
		t.Errorf("stats.OpenConnections = %d; want 0", got)
	}
}

func TestConnMaxLifetime(t *testing.T) {
	t0 := time.Unix(1000000, 0)
	offset := time.Duration(0)

	nowFunc = func { t0.Add(offset) }
	defer func() { nowFunc = time.Now }()

	db := newTestDB(t, "magicquery")
	defer closeDB(t, db)

	driver := db.Driver().(*fakeDriver)

	// Force the number of open connections to 0 so we can get an accurate
	// count for the test
	db.clearAllConns(t)

	driver.mu.Lock()
	opens0 := driver.openCount
	closes0 := driver.closeCount
	driver.mu.Unlock()

	db.SetMaxIdleConns(10)
	db.SetMaxOpenConns(10)

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}

	offset = time.Second
	tx2, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}

	tx.Commit()
	tx2.Commit()

	driver.mu.Lock()
	opens := driver.openCount - opens0
	closes := driver.closeCount - closes0
	driver.mu.Unlock()

	if opens != 2 {
		t.Errorf("opens = %d; want 2", opens)
	}
	if closes != 0 {
		t.Errorf("closes = %d; want 0", closes)
	}
	if g, w := db.numFreeConns(), 2; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	// Expire first conn
	offset = 11 * time.Second
	db.SetConnMaxLifetime(10 * time.Second)

	tx, err = db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	tx2, err = db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	tx.Commit()
	tx2.Commit()

	// Give connectionCleaner chance to run.
	waitCondition(t, func {
		driver.mu.Lock()
		opens = driver.openCount - opens0
		closes = driver.closeCount - closes0
		driver.mu.Unlock()

		return closes == 1
	})

	if opens != 3 {
		t.Errorf("opens = %d; want 3", opens)
	}
	if closes != 1 {
		t.Errorf("closes = %d; want 1", closes)
	}

	if s := db.Stats(); s.MaxLifetimeClosed != 1 {
		t.Errorf("MaxLifetimeClosed = %d; want 1 %#v", s.MaxLifetimeClosed, s)
	}
}

// golang.org/issue/5323
func TestStmtCloseDeps(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	defer setHookpostCloseConn(nil)
	setHookpostCloseConn(func { _, err -> if err != nil {
		t.Errorf("Error closing fakeConn: %v", err)
	} })

	db := newTestDB(t, "magicquery")
	defer closeDB(t, db)

	driver := db.Driver().(*fakeDriver)

	driver.mu.Lock()
	opens0 := driver.openCount
	closes0 := driver.closeCount
	driver.mu.Unlock()
	openDelta0 := opens0 - closes0

	stmt, err := db.Prepare("SELECT|magicquery|op|op=?,millis=?")
	if err != nil {
		t.Fatal(err)
	}

	// Start 50 parallel slow queries.
	const (
		nquery      = 50
		sleepMillis = 25
		nbatch      = 2
	)
	var wg sync.WaitGroup
	for batch := 0; batch < nbatch; batch++ {
		for i := 0; i < nquery; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				var op string
				if err := stmt.QueryRow("sleep", sleepMillis).Scan(&op); err != nil && err != ErrNoRows {
					t.Error(err)
				}
			}()
		}
		// Wait for the batch of queries above to finish before starting the next round.
		wg.Wait()
	}

	if g, w := db.numFreeConns(), 2; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	if n := db.numDepsPoll(t, 4); n > 4 {
		t.Errorf("number of dependencies = %d; expected <= 4", n)
		db.dumpDeps(t)
	}

	driver.mu.Lock()
	opens := driver.openCount - opens0
	closes := driver.closeCount - closes0
	openDelta := (driver.openCount - driver.closeCount) - openDelta0
	driver.mu.Unlock()

	if openDelta > 2 {
		t.Logf("open calls = %d", opens)
		t.Logf("close calls = %d", closes)
		t.Logf("open delta = %d", openDelta)
		t.Errorf("db connections opened = %d; want <= 2", openDelta)
		db.dumpDeps(t)
	}

	if !waitCondition(t, func { len(stmt.css) <= nquery }) {
		t.Errorf("len(stmt.css) = %d; want <= %d", len(stmt.css), nquery)
	}

	if err := stmt.Close(); err != nil {
		t.Fatal(err)
	}

	if g, w := db.numFreeConns(), 2; g != w {
		t.Errorf("free conns = %d; want %d", g, w)
	}

	if n := db.numDepsPoll(t, 2); n > 2 {
		t.Errorf("number of dependencies = %d; expected <= 2", n)
		db.dumpDeps(t)
	}

	db.clearAllConns(t)
}

// golang.org/issue/5046
func TestCloseConnBeforeStmts(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	defer setHookpostCloseConn(nil)
	setHookpostCloseConn(func { _, err ->
		if err != nil {
			t.Errorf("Error closing fakeConn: %v; from %s", err, stack())
			db.dumpDeps(t)
			t.Errorf("DB = %#v", db)
		}
	})

	stmt, err := db.Prepare("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}

	if len(db.freeConn) != 1 {
		t.Fatalf("expected 1 freeConn; got %d", len(db.freeConn))
	}
	dc := db.freeConn[0]
	if dc.closed {
		t.Errorf("conn shouldn't be closed")
	}

	if n := len(dc.openStmt); n != 1 {
		t.Errorf("driverConn num openStmt = %d; want 1", n)
	}
	err = db.Close()
	if err != nil {
		t.Errorf("db Close = %v", err)
	}
	if !dc.closed {
		t.Errorf("after db.Close, driverConn should be closed")
	}
	if n := len(dc.openStmt); n != 0 {
		t.Errorf("driverConn num openStmt = %d; want 0", n)
	}

	err = stmt.Close()
	if err != nil {
		t.Errorf("Stmt close = %v", err)
	}

	if !dc.closed {
		t.Errorf("conn should be closed")
	}
	if dc.ci != nil {
		t.Errorf("after Stmt Close, driverConn's Conn interface should be nil")
	}
}

// golang.org/issue/5283: don't release the Rows' connection in Close
// before calling Stmt.Close.
func TestRowsCloseOrder(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	db.SetMaxIdleConns(0)
	setStrictFakeConnClose(t)
	defer setStrictFakeConnClose(nil)

	rows, err := db.Query("SELECT|people|age,name|")
	if err != nil {
		t.Fatal(err)
	}
	err = rows.Close()
	if err != nil {
		t.Fatal(err)
	}
}

func TestRowsImplicitClose(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	rows, err := db.Query("SELECT|people|age,name|")
	if err != nil {
		t.Fatal(err)
	}

	want, fail := 2, errors.New("fail")
	r := rows.rowsi.(*rowsCursor)
	r.errPos, r.err = want, fail

	got := 0
	for rows.Next() {
		got++
	}
	if got != want {
		t.Errorf("got %d rows, want %d", got, want)
	}
	if err := rows.Err(); err != fail {
		t.Errorf("got error %v, want %v", err, fail)
	}
	if !r.closed {
		t.Errorf("r.closed is false, want true")
	}
}

func TestRowsCloseError(t *testing.T) {
	db := newTestDB(t, "people")
	defer db.Close()
	rows, err := db.Query("SELECT|people|age,name|")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	type row struct {
		age  int
		name string
	}
	got := []row{}

	rc, ok := rows.rowsi.(*rowsCursor)
	if !ok {
		t.Fatal("not using *rowsCursor")
	}
	rc.closeErr = errors.New("rowsCursor: failed to close")

	for rows.Next() {
		var r row
		err = rows.Scan(&r.age, &r.name)
		if err != nil {
			t.Fatalf("Scan: %v", err)
		}
		got = append(got, r)
	}
	err = rows.Err()
	if err != rc.closeErr {
		t.Fatalf("unexpected err: got %v, want %v", err, rc.closeErr)
	}
}

func TestStmtCloseOrder(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	db.SetMaxIdleConns(0)
	setStrictFakeConnClose(t)
	defer setStrictFakeConnClose(nil)

	_, err := db.Query("SELECT|non_existent|name|")
	if err == nil {
		t.Fatal("Querying non-existent table should fail")
	}
}

// Test cases where there's more than maxBadConnRetries bad connections in the
// pool (issue 8834)
func TestManyErrBadConn(t *testing.T) {
	manyErrBadConnSetup := func(first ...func(db *DB)) *DB {
		db := newTestDB(t, "people")

		for _, f := range first {
			f(db)
		}

		nconn := maxBadConnRetries + 1
		db.SetMaxIdleConns(nconn)
		db.SetMaxOpenConns(nconn)
		// open enough connections
		func() {
			for i := 0; i < nconn; i++ {
				rows, err := db.Query("SELECT|people|age,name|")
				if err != nil {
					t.Fatal(err)
				}
				defer rows.Close()
			}
		}()

		db.mu.Lock()
		defer db.mu.Unlock()
		if db.numOpen != nconn {
			t.Fatalf("unexpected numOpen %d (was expecting %d)", db.numOpen, nconn)
		} else if len(db.freeConn) != nconn {
			t.Fatalf("unexpected len(db.freeConn) %d (was expecting %d)", len(db.freeConn), nconn)
		}
		for _, conn := range db.freeConn {
			conn.Lock()
			conn.ci.(*fakeConn).stickyBad = true
			conn.Unlock()
		}
		return db
	}

	// Query
	db := manyErrBadConnSetup()
	defer closeDB(t, db)
	rows, err := db.Query("SELECT|people|age,name|")
	if err != nil {
		t.Fatal(err)
	}
	if err = rows.Close(); err != nil {
		t.Fatal(err)
	}

	// Exec
	db = manyErrBadConnSetup()
	defer closeDB(t, db)
	_, err = db.Exec("INSERT|people|name=Julia,age=19")
	if err != nil {
		t.Fatal(err)
	}

	// Begin
	db = manyErrBadConnSetup()
	defer closeDB(t, db)
	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	if err = tx.Rollback(); err != nil {
		t.Fatal(err)
	}

	// Prepare
	db = manyErrBadConnSetup()
	defer closeDB(t, db)
	stmt, err := db.Prepare("SELECT|people|age,name|")
	if err != nil {
		t.Fatal(err)
	}
	if err = stmt.Close(); err != nil {
		t.Fatal(err)
	}

	// Stmt.Exec
	db = manyErrBadConnSetup(func { db ->
		stmt, err = db.Prepare("INSERT|people|name=Julia,age=19")
		if err != nil {
			t.Fatal(err)
		}
	})
	defer closeDB(t, db)
	_, err = stmt.Exec()
	if err != nil {
		t.Fatal(err)
	}
	if err = stmt.Close(); err != nil {
		t.Fatal(err)
	}

	// Stmt.Query
	db = manyErrBadConnSetup(func { db ->
		stmt, err = db.Prepare("SELECT|people|age,name|")
		if err != nil {
			t.Fatal(err)
		}
	})
	defer closeDB(t, db)
	rows, err = stmt.Query()
	if err != nil {
		t.Fatal(err)
	}
	if err = rows.Close(); err != nil {
		t.Fatal(err)
	}
	if err = stmt.Close(); err != nil {
		t.Fatal(err)
	}

	// Conn
	db = manyErrBadConnSetup()
	defer closeDB(t, db)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	conn, err := db.Conn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	conn.dc.ci.(*fakeConn).skipDirtySession = true
	err = conn.Close()
	if err != nil {
		t.Fatal(err)
	}

	// Ping
	db = manyErrBadConnSetup()
	defer closeDB(t, db)
	err = db.PingContext(ctx)
	if err != nil {
		t.Fatal(err)
	}
}

// Issue 34775: Ensure that a Tx cannot commit after a rollback.
func TestTxCannotCommitAfterRollback(t *testing.T) {
	db := newTestDB(t, "tx_status")
	defer closeDB(t, db)

	// First check query reporting is correct.
	var txStatus string
	err := db.QueryRow("SELECT|tx_status|tx_status|").Scan(&txStatus)
	if err != nil {
		t.Fatal(err)
	}
	if g, w := txStatus, "autocommit"; g != w {
		t.Fatalf("tx_status=%q, wanted %q", g, w)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Ignore dirty session for this test.
	// A failing test should trigger the dirty session flag as well,
	// but that isn't exactly what this should test for.
	tx.txi.(*fakeTx).c.skipDirtySession = true

	defer tx.Rollback()

	err = tx.QueryRow("SELECT|tx_status|tx_status|").Scan(&txStatus)
	if err != nil {
		t.Fatal(err)
	}
	if g, w := txStatus, "transaction"; g != w {
		t.Fatalf("tx_status=%q, wanted %q", g, w)
	}

	// 1. Begin a transaction.
	// 2. (A) Start a query, (B) begin Tx rollback through a ctx cancel.
	// 3. Check if 2.A has committed in Tx (pass) or outside of Tx (fail).
	sendQuery := make(chan struct{})
	// The Tx status is returned through the row results, ensure
	// that the rows results are not canceled.
	bypassRowsAwaitDone = true
	hookTxGrabConn = func() {
		cancel()
		<-sendQuery
	}
	rollbackHook = func() {
		close(sendQuery)
	}
	defer func() {
		hookTxGrabConn = nil
		rollbackHook = nil
		bypassRowsAwaitDone = false
	}()

	err = tx.QueryRow("SELECT|tx_status|tx_status|").Scan(&txStatus)
	if err != nil {
		// A failure here would be expected if skipDirtySession was not set to true above.
		t.Fatal(err)
	}
	if g, w := txStatus, "transaction"; g != w {
		t.Fatalf("tx_status=%q, wanted %q", g, w)
	}
}

// Issue 40985 transaction statement deadlock while context cancel.
func TestTxStmtDeadlock(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}

	stmt, err := tx.Prepare("SELECT|people|name,age|age=?")
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	// Run number of stmt queries to reproduce deadlock from context cancel
	for i := 0; i < 1e3; i++ {
		// Encounter any close related errors (e.g. ErrTxDone, stmt is closed)
		// is expected due to context cancel.
		_, err = stmt.Query(1)
		if err != nil {
			break
		}
	}
	_ = tx.Rollback()
}

// Issue32530 encounters an issue where a connection may
// expire right after it comes out of a used connection pool
// even when a new connection is requested.
func TestConnExpiresFreshOutOfPool(t *testing.T) {
	execCases := []struct {
		expired  bool
		badReset bool
	}{
		{false, false},
		{true, false},
		{false, true},
	}

	t0 := time.Unix(1000000, 0)
	offset := time.Duration(0)
	offsetMu := sync.RWMutex{}

	nowFunc = func {
		offsetMu.RLock()
		defer offsetMu.RUnlock()
		return t0.Add(offset)
	}
	defer func() { nowFunc = time.Now }()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	db := newTestDB(t, "magicquery")
	defer closeDB(t, db)

	db.SetMaxOpenConns(1)

	for _, ec := range execCases {
		ec := ec
		name := fmt.Sprintf("expired=%t,badReset=%t", ec.expired, ec.badReset)
		t.Run(name, func { t ->
			db.clearAllConns(t)

			db.SetMaxIdleConns(1)
			db.SetConnMaxLifetime(10 * time.Second)

			conn, err := db.conn(ctx, alwaysNewConn)
			if err != nil {
				t.Fatal(err)
			}

			afterPutConn := make(chan struct{})
			waitingForConn := make(chan struct{})

			go func() {
				defer close(afterPutConn)

				conn, err := db.conn(ctx, alwaysNewConn)
				if err == nil {
					db.putConn(conn, err, false)
				} else {
					t.Errorf("db.conn: %v", err)
				}
			}()
			go func() {
				defer close(waitingForConn)

				for {
					if t.Failed() {
						return
					}
					db.mu.Lock()
					ct := db.connRequests.Len()
					db.mu.Unlock()
					if ct > 0 {
						return
					}
					time.Sleep(pollDuration)
				}
			}()

			<-waitingForConn

			if t.Failed() {
				return
			}

			offsetMu.Lock()
			if ec.expired {
				offset = 11 * time.Second
			} else {
				offset = time.Duration(0)
			}
			offsetMu.Unlock()

			conn.ci.(*fakeConn).stickyBad = ec.badReset

			db.putConn(conn, err, true)

			<-afterPutConn
		})
	}
}

// TestIssue20575 ensures the Rows from query does not block
// closing a transaction. Ensure Rows is closed while closing a transaction.
func TestIssue20575(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, err = tx.QueryContext(ctx, "SELECT|people|age,name|")
	if err != nil {
		t.Fatal(err)
	}
	// Do not close Rows from QueryContext.
	err = tx.Rollback()
	if err != nil {
		t.Fatal(err)
	}
	select {
	default:
	case <-ctx.Done():
		t.Fatal("timeout: failed to rollback query without closing rows:", ctx.Err())
	}
}

// TestIssue20622 tests closing the transaction before rows is closed, requires
// the race detector to fail.
func TestIssue20622(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}

	rows, err := tx.Query("SELECT|people|age,name|")
	if err != nil {
		t.Fatal(err)
	}

	count := 0
	for rows.Next() {
		count++
		var age int
		var name string
		if err := rows.Scan(&age, &name); err != nil {
			t.Fatal("scan failed", err)
		}

		if count == 1 {
			cancel()
		}
		time.Sleep(100 * time.Millisecond)
	}
	rows.Close()
	tx.Commit()
}

// golang.org/issue/5718
func TestErrBadConnReconnect(t *testing.T) {
	db := newTestDB(t, "foo")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32,dead=bool")

	simulateBadConn := func(name string, hook *func() bool, op func() error) {
		broken, retried := false, false
		numOpen := db.numOpen

		// simulate a broken connection on the first try
		*hook = func {
			if !broken {
				broken = true
				return true
			}
			retried = true
			return false
		}

		if err := op(); err != nil {
			t.Errorf(name+": %v", err)
			return
		}

		if !broken || !retried {
			t.Error(name + ": Failed to simulate broken connection")
		}
		*hook = nil

		if numOpen != db.numOpen {
			t.Errorf(name+": leaked %d connection(s)!", db.numOpen-numOpen)
			numOpen = db.numOpen
		}
	}

	// db.Exec
	dbExec := func {
		_, err := db.Exec("INSERT|t1|name=?,age=?,dead=?", "Gordon", 3, true)
		return err
	}
	simulateBadConn("db.Exec prepare", &hookPrepareBadConn, dbExec)
	simulateBadConn("db.Exec exec", &hookExecBadConn, dbExec)

	// db.Query
	dbQuery := func {
		rows, err := db.Query("SELECT|t1|age,name|")
		if err == nil {
			err = rows.Close()
		}
		return err
	}
	simulateBadConn("db.Query prepare", &hookPrepareBadConn, dbQuery)
	simulateBadConn("db.Query query", &hookQueryBadConn, dbQuery)

	// db.Prepare
	simulateBadConn("db.Prepare", &hookPrepareBadConn, func {
		stmt, err := db.Prepare("INSERT|t1|name=?,age=?,dead=?")
		if err != nil {
			return err
		}
		stmt.Close()
		return nil
	})

	// Provide a way to force a re-prepare of a statement on next execution
	forcePrepare := func(stmt *Stmt) {
		stmt.css = nil
	}

	// stmt.Exec
	stmt1, err := db.Prepare("INSERT|t1|name=?,age=?,dead=?")
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	defer stmt1.Close()
	// make sure we must prepare the stmt first
	forcePrepare(stmt1)

	stmtExec := func {
		_, err := stmt1.Exec("Gopher", 3, false)
		return err
	}
	simulateBadConn("stmt.Exec prepare", &hookPrepareBadConn, stmtExec)
	simulateBadConn("stmt.Exec exec", &hookExecBadConn, stmtExec)

	// stmt.Query
	stmt2, err := db.Prepare("SELECT|t1|age,name|")
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	defer stmt2.Close()
	// make sure we must prepare the stmt first
	forcePrepare(stmt2)

	stmtQuery := func {
		rows, err := stmt2.Query()
		if err == nil {
			err = rows.Close()
		}
		return err
	}
	simulateBadConn("stmt.Query prepare", &hookPrepareBadConn, stmtQuery)
	simulateBadConn("stmt.Query exec", &hookQueryBadConn, stmtQuery)
}

// golang.org/issue/11264
func TestTxEndBadConn(t *testing.T) {
	db := newTestDB(t, "foo")
	defer closeDB(t, db)
	db.SetMaxIdleConns(0)
	exec(t, db, "CREATE|t1|name=string,age=int32,dead=bool")
	db.SetMaxIdleConns(1)

	simulateBadConn := func(name string, hook *func() bool, op func() error) {
		broken := false
		numOpen := db.numOpen

		*hook = func {
			if !broken {
				broken = true
			}
			return broken
		}

		if err := op(); !errors.Is(err, driver.ErrBadConn) {
			t.Errorf(name+": %v", err)
			return
		}

		if !broken {
			t.Error(name + ": Failed to simulate broken connection")
		}
		*hook = nil

		if numOpen != db.numOpen {
			t.Errorf(name+": leaked %d connection(s)!", db.numOpen-numOpen)
		}
	}

	// db.Exec
	dbExec := func(endTx func(tx *Tx) error) func() error {
		return func() error {
			tx, err := db.Begin()
			if err != nil {
				return err
			}
			_, err = tx.Exec("INSERT|t1|name=?,age=?,dead=?", "Gordon", 3, true)
			if err != nil {
				return err
			}
			return endTx(tx)
		}
	}
	simulateBadConn("db.Tx.Exec commit", &hookCommitBadConn, dbExec((*Tx).Commit))
	simulateBadConn("db.Tx.Exec rollback", &hookRollbackBadConn, dbExec((*Tx).Rollback))

	// db.Query
	dbQuery := func(endTx func(tx *Tx) error) func() error {
		return func() error {
			tx, err := db.Begin()
			if err != nil {
				return err
			}
			rows, err := tx.Query("SELECT|t1|age,name|")
			if err == nil {
				err = rows.Close()
			} else {
				return err
			}
			return endTx(tx)
		}
	}
	simulateBadConn("db.Tx.Query commit", &hookCommitBadConn, dbQuery((*Tx).Commit))
	simulateBadConn("db.Tx.Query rollback", &hookRollbackBadConn, dbQuery((*Tx).Rollback))
}

type concurrentTest interface {
	init(t testing.TB, db *DB)
	finish(t testing.TB)
	test(t testing.TB) error
}

type concurrentDBQueryTest struct {
	db *DB
}

func (c *concurrentDBQueryTest) init(t testing.TB, db *DB) {
	c.db = db
}

func (c *concurrentDBQueryTest) finish(t testing.TB) {
	c.db = nil
}

func (c *concurrentDBQueryTest) test(t testing.TB) error {
	rows, err := c.db.Query("SELECT|people|name|")
	if err != nil {
		t.Error(err)
		return err
	}
	var name string
	for rows.Next() {
		rows.Scan(&name)
	}
	rows.Close()
	return nil
}

type concurrentDBExecTest struct {
	db *DB
}

func (c *concurrentDBExecTest) init(t testing.TB, db *DB) {
	c.db = db
}

func (c *concurrentDBExecTest) finish(t testing.TB) {
	c.db = nil
}

func (c *concurrentDBExecTest) test(t testing.TB) error {
	_, err := c.db.Exec("NOSERT|people|name=Chris,age=?,photo=CPHOTO,bdate=?", 3, chrisBirthday)
	if err != nil {
		t.Error(err)
		return err
	}
	return nil
}

type concurrentStmtQueryTest struct {
	db   *DB
	stmt *Stmt
}

func (c *concurrentStmtQueryTest) init(t testing.TB, db *DB) {
	c.db = db
	var err error
	c.stmt, err = db.Prepare("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
}

func (c *concurrentStmtQueryTest) finish(t testing.TB) {
	if c.stmt != nil {
		c.stmt.Close()
		c.stmt = nil
	}
	c.db = nil
}

func (c *concurrentStmtQueryTest) test(t testing.TB) error {
	rows, err := c.stmt.Query()
	if err != nil {
		t.Errorf("error on query:  %v", err)
		return err
	}

	var name string
	for rows.Next() {
		rows.Scan(&name)
	}
	rows.Close()
	return nil
}

type concurrentStmtExecTest struct {
	db   *DB
	stmt *Stmt
}

func (c *concurrentStmtExecTest) init(t testing.TB, db *DB) {
	c.db = db
	var err error
	c.stmt, err = db.Prepare("NOSERT|people|name=Chris,age=?,photo=CPHOTO,bdate=?")
	if err != nil {
		t.Fatal(err)
	}
}

func (c *concurrentStmtExecTest) finish(t testing.TB) {
	if c.stmt != nil {
		c.stmt.Close()
		c.stmt = nil
	}
	c.db = nil
}

func (c *concurrentStmtExecTest) test(t testing.TB) error {
	_, err := c.stmt.Exec(3, chrisBirthday)
	if err != nil {
		t.Errorf("error on exec:  %v", err)
		return err
	}
	return nil
}

type concurrentTxQueryTest struct {
	db *DB
	tx *Tx
}

func (c *concurrentTxQueryTest) init(t testing.TB, db *DB) {
	c.db = db
	var err error
	c.tx, err = c.db.Begin()
	if err != nil {
		t.Fatal(err)
	}
}

func (c *concurrentTxQueryTest) finish(t testing.TB) {
	if c.tx != nil {
		c.tx.Rollback()
		c.tx = nil
	}
	c.db = nil
}

func (c *concurrentTxQueryTest) test(t testing.TB) error {
	rows, err := c.db.Query("SELECT|people|name|")
	if err != nil {
		t.Error(err)
		return err
	}
	var name string
	for rows.Next() {
		rows.Scan(&name)
	}
	rows.Close()
	return nil
}

type concurrentTxExecTest struct {
	db *DB
	tx *Tx
}

func (c *concurrentTxExecTest) init(t testing.TB, db *DB) {
	c.db = db
	var err error
	c.tx, err = c.db.Begin()
	if err != nil {
		t.Fatal(err)
	}
}

func (c *concurrentTxExecTest) finish(t testing.TB) {
	if c.tx != nil {
		c.tx.Rollback()
		c.tx = nil
	}
	c.db = nil
}

func (c *concurrentTxExecTest) test(t testing.TB) error {
	_, err := c.tx.Exec("NOSERT|people|name=Chris,age=?,photo=CPHOTO,bdate=?", 3, chrisBirthday)
	if err != nil {
		t.Error(err)
		return err
	}
	return nil
}

type concurrentTxStmtQueryTest struct {
	db   *DB
	tx   *Tx
	stmt *Stmt
}

func (c *concurrentTxStmtQueryTest) init(t testing.TB, db *DB) {
	c.db = db
	var err error
	c.tx, err = c.db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	c.stmt, err = c.tx.Prepare("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
}

func (c *concurrentTxStmtQueryTest) finish(t testing.TB) {
	if c.stmt != nil {
		c.stmt.Close()
		c.stmt = nil
	}
	if c.tx != nil {
		c.tx.Rollback()
		c.tx = nil
	}
	c.db = nil
}

func (c *concurrentTxStmtQueryTest) test(t testing.TB) error {
	rows, err := c.stmt.Query()
	if err != nil {
		t.Errorf("error on query:  %v", err)
		return err
	}

	var name string
	for rows.Next() {
		rows.Scan(&name)
	}
	rows.Close()
	return nil
}

type concurrentTxStmtExecTest struct {
	db   *DB
	tx   *Tx
	stmt *Stmt
}

func (c *concurrentTxStmtExecTest) init(t testing.TB, db *DB) {
	c.db = db
	var err error
	c.tx, err = c.db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	c.stmt, err = c.tx.Prepare("NOSERT|people|name=Chris,age=?,photo=CPHOTO,bdate=?")
	if err != nil {
		t.Fatal(err)
	}
}

func (c *concurrentTxStmtExecTest) finish(t testing.TB) {
	if c.stmt != nil {
		c.stmt.Close()
		c.stmt = nil
	}
	if c.tx != nil {
		c.tx.Rollback()
		c.tx = nil
	}
	c.db = nil
}

func (c *concurrentTxStmtExecTest) test(t testing.TB) error {
	_, err := c.stmt.Exec(3, chrisBirthday)
	if err != nil {
		t.Errorf("error on exec:  %v", err)
		return err
	}
	return nil
}

type concurrentRandomTest struct {
	tests []concurrentTest
}

func (c *concurrentRandomTest) init(t testing.TB, db *DB) {
	c.tests = []concurrentTest{
		new(concurrentDBQueryTest),
		new(concurrentDBExecTest),
		new(concurrentStmtQueryTest),
		new(concurrentStmtExecTest),
		new(concurrentTxQueryTest),
		new(concurrentTxExecTest),
		new(concurrentTxStmtQueryTest),
		new(concurrentTxStmtExecTest),
	}
	for _, ct := range c.tests {
		ct.init(t, db)
	}
}

func (c *concurrentRandomTest) finish(t testing.TB) {
	for _, ct := range c.tests {
		ct.finish(t)
	}
}

func (c *concurrentRandomTest) test(t testing.TB) error {
	ct := c.tests[rand.Intn(len(c.tests))]
	return ct.test(t)
}

func doConcurrentTest(t testing.TB, ct concurrentTest) {
	maxProcs, numReqs := 1, 500
	if testing.Short() {
		maxProcs, numReqs = 4, 50
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(maxProcs))

	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ct.init(t, db)
	defer ct.finish(t)

	var wg sync.WaitGroup
	wg.Add(numReqs)

	reqs := make(chan bool)
	defer close(reqs)

	for i := 0; i < maxProcs*2; i++ {
		go func() {
			for range reqs {
				err := ct.test(t)
				if err != nil {
					wg.Done()
					continue
				}
				wg.Done()
			}
		}()
	}

	for i := 0; i < numReqs; i++ {
		reqs <- true
	}

	wg.Wait()
}

func TestIssue6081(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	drv := db.Driver().(*fakeDriver)
	drv.mu.Lock()
	opens0 := drv.openCount
	closes0 := drv.closeCount
	drv.mu.Unlock()

	stmt, err := db.Prepare("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	setRowsCloseHook(func { rows, err -> *err = driver.ErrBadConn })
	defer setRowsCloseHook(nil)
	for i := 0; i < 10; i++ {
		rows, err := stmt.Query()
		if err != nil {
			t.Fatal(err)
		}
		rows.Close()
	}
	if n := len(stmt.css); n > 1 {
		t.Errorf("len(css slice) = %d; want <= 1", n)
	}
	stmt.Close()
	if n := len(stmt.css); n != 0 {
		t.Errorf("len(css slice) after Close = %d; want 0", n)
	}

	drv.mu.Lock()
	opens := drv.openCount - opens0
	closes := drv.closeCount - closes0
	drv.mu.Unlock()
	if opens < 9 {
		t.Errorf("opens = %d; want >= 9", opens)
	}
	if closes < 9 {
		t.Errorf("closes = %d; want >= 9", closes)
	}
}

// TestIssue18429 attempts to stress rolling back the transaction from a
// context cancel while simultaneously calling Tx.Rollback. Rolling back from a
// context happens concurrently so tx.rollback and tx.Commit must guard against
// double entry.
//
// In the test, a context is canceled while the query is in process so
// the internal rollback will run concurrently with the explicitly called
// Tx.Rollback.
//
// The addition of calling rows.Next also tests
// Issue 21117.
func TestIssue18429(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx := context.Background()
	sem := make(chan bool, 20)
	var wg sync.WaitGroup

	const milliWait = 30

	for i := 0; i < 100; i++ {
		sem <- true
		wg.Add(1)
		go func() {
			defer func() {
				<-sem
				wg.Done()
			}()
			qwait := (time.Duration(rand.Intn(milliWait)) * time.Millisecond).String()

			ctx, cancel := context.WithTimeout(ctx, time.Duration(rand.Intn(milliWait))*time.Millisecond)
			defer cancel()

			tx, err := db.BeginTx(ctx, nil)
			if err != nil {
				return
			}
			// This is expected to give a cancel error most, but not all the time.
			// Test failure will happen with a panic or other race condition being
			// reported.
			rows, _ := tx.QueryContext(ctx, "WAIT|"+qwait+"|SELECT|people|name|")
			if rows != nil {
				var name string
				// Call Next to test Issue 21117 and check for races.
				for rows.Next() {
					// Scan the buffer so it is read and checked for races.
					rows.Scan(&name)
				}
				rows.Close()
			}
			// This call will race with the context cancel rollback to complete
			// if the rollback itself isn't guarded.
			tx.Rollback()
		}()
	}
	wg.Wait()
}

// TestIssue20160 attempts to test a short context life on a stmt Query.
func TestIssue20160(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx := context.Background()
	sem := make(chan bool, 20)
	var wg sync.WaitGroup

	const milliWait = 30

	stmt, err := db.PrepareContext(ctx, "SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	defer stmt.Close()

	for i := 0; i < 100; i++ {
		sem <- true
		wg.Add(1)
		go func() {
			defer func() {
				<-sem
				wg.Done()
			}()
			ctx, cancel := context.WithTimeout(ctx, time.Duration(rand.Intn(milliWait))*time.Millisecond)
			defer cancel()

			// This is expected to give a cancel error most, but not all the time.
			// Test failure will happen with a panic or other race condition being
			// reported.
			rows, _ := stmt.QueryContext(ctx)
			if rows != nil {
				rows.Close()
			}
		}()
	}
	wg.Wait()
}

// TestIssue18719 closes the context right before use. The sql.driverConn
// will nil out the ci on close in a lock, but if another process uses it right after
// it will panic with on the nil ref.
//
// See https://golang.org/cl/35550 .
func TestIssue18719(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}

	hookTxGrabConn = func() {
		cancel()

		// Wait for the context to cancel and tx to rollback.
		for !tx.isDone() {
			time.Sleep(pollDuration)
		}
	}
	defer func() { hookTxGrabConn = nil }()

	// This call will grab the connection and cancel the context
	// after it has done so. Code after must deal with the canceled state.
	_, err = tx.QueryContext(ctx, "SELECT|people|name|")
	if err != nil {
		t.Fatalf("expected error %v but got %v", nil, err)
	}

	// Rows may be ignored because it will be closed when the context is canceled.

	// Do not explicitly rollback. The rollback will happen from the
	// canceled context.

	cancel()
}

func TestIssue20647(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	conn, err := db.Conn(ctx)
	if err != nil {
		t.Fatal(err)
	}
	conn.dc.ci.(*fakeConn).skipDirtySession = true
	defer conn.Close()

	stmt, err := conn.PrepareContext(ctx, "SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	defer stmt.Close()

	rows1, err := stmt.QueryContext(ctx)
	if err != nil {
		t.Fatal("rows1", err)
	}
	defer rows1.Close()

	rows2, err := stmt.QueryContext(ctx)
	if err != nil {
		t.Fatal("rows2", err)
	}
	defer rows2.Close()

	if rows1.dc != rows2.dc {
		t.Fatal("stmt prepared on Conn does not use same connection")
	}
}

func TestConcurrency(t *testing.T) {
	list := []struct {
		name string
		ct   concurrentTest
	}{
		{"Query", new(concurrentDBQueryTest)},
		{"Exec", new(concurrentDBExecTest)},
		{"StmtQuery", new(concurrentStmtQueryTest)},
		{"StmtExec", new(concurrentStmtExecTest)},
		{"TxQuery", new(concurrentTxQueryTest)},
		{"TxExec", new(concurrentTxExecTest)},
		{"TxStmtQuery", new(concurrentTxStmtQueryTest)},
		{"TxStmtExec", new(concurrentTxStmtExecTest)},
		{"Random", new(concurrentRandomTest)},
	}
	for _, item := range list {
		t.Run(item.name, func { t -> doConcurrentTest(t, item.ct) })
	}
}

func TestConnectionLeak(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	// Start by opening defaultMaxIdleConns
	rows := make([]*Rows, defaultMaxIdleConns)
	// We need to SetMaxOpenConns > MaxIdleConns, so the DB can open
	// a new connection and we can fill the idle queue with the released
	// connections.
	db.SetMaxOpenConns(len(rows) + 1)
	for ii := range rows {
		r, err := db.Query("SELECT|people|name|")
		if err != nil {
			t.Fatal(err)
		}
		r.Next()
		if err := r.Err(); err != nil {
			t.Fatal(err)
		}
		rows[ii] = r
	}
	// Now we have defaultMaxIdleConns busy connections. Open
	// a new one, but wait until the busy connections are released
	// before returning control to DB.
	drv := db.Driver().(*fakeDriver)
	drv.waitCh = make(chan struct{}, 1)
	drv.waitingCh = make(chan struct{}, 1)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		r, err := db.Query("SELECT|people|name|")
		if err != nil {
			t.Error(err)
			return
		}
		r.Close()
		wg.Done()
	}()
	// Wait until the goroutine we've just created has started waiting.
	<-drv.waitingCh
	// Now close the busy connections. This provides a connection for
	// the blocked goroutine and then fills up the idle queue.
	for _, v := range rows {
		v.Close()
	}
	// At this point we give the new connection to DB. This connection is
	// now useless, since the idle queue is full and there are no pending
	// requests. DB should deal with this situation without leaking the
	// connection.
	drv.waitCh <- struct{}{}
	wg.Wait()
}

func TestStatsMaxIdleClosedZero(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	db.SetConnMaxLifetime(0)

	preMaxIdleClosed := db.Stats().MaxIdleClosed

	for i := 0; i < 10; i++ {
		rows, err := db.Query("SELECT|people|name|")
		if err != nil {
			t.Fatal(err)
		}
		rows.Close()
	}

	st := db.Stats()
	maxIdleClosed := st.MaxIdleClosed - preMaxIdleClosed
	t.Logf("MaxIdleClosed: %d", maxIdleClosed)
	if maxIdleClosed != 0 {
		t.Fatal("expected 0 max idle closed conns, got: ", maxIdleClosed)
	}
}

func TestStatsMaxIdleClosedTen(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(0)
	db.SetConnMaxLifetime(0)

	preMaxIdleClosed := db.Stats().MaxIdleClosed

	for i := 0; i < 10; i++ {
		rows, err := db.Query("SELECT|people|name|")
		if err != nil {
			t.Fatal(err)
		}
		rows.Close()
	}

	st := db.Stats()
	maxIdleClosed := st.MaxIdleClosed - preMaxIdleClosed
	t.Logf("MaxIdleClosed: %d", maxIdleClosed)
	if maxIdleClosed != 10 {
		t.Fatal("expected 0 max idle closed conns, got: ", maxIdleClosed)
	}
}

// testUseConns uses count concurrent connections with 1 nanosecond apart.
// Returns the returnedAt time of the final connection.
func testUseConns(t *testing.T, count int, tm time.Time, db *DB) time.Time {
	conns := make([]*Conn, count)
	ctx := context.Background()
	for i := range conns {
		tm = tm.Add(time.Nanosecond)
		nowFunc = func { tm }
		c, err := db.Conn(ctx)
		if err != nil {
			t.Error(err)
		}
		conns[i] = c
	}

	for i := len(conns) - 1; i >= 0; i-- {
		tm = tm.Add(time.Nanosecond)
		nowFunc = func { tm }
		if err := conns[i].Close(); err != nil {
			t.Error(err)
		}
	}

	return tm
}

func TestMaxIdleTime(t *testing.T) {
	usedConns := 5
	reusedConns := 2
	list := []struct {
		wantMaxIdleTime   time.Duration
		wantMaxLifetime   time.Duration
		wantNextCheck     time.Duration
		wantIdleClosed    int64
		wantMaxIdleClosed int64
		timeOffset        time.Duration
		secondTimeOffset  time.Duration
	}{
		{
			time.Millisecond,
			0,
			time.Millisecond - time.Nanosecond,
			int64(usedConns - reusedConns),
			int64(usedConns - reusedConns),
			10 * time.Millisecond,
			0,
		},
		{
			// Want to close some connections via max idle time and one by max lifetime.
			time.Millisecond,
			// nowFunc() - MaxLifetime should be 1 * time.Nanosecond in connectionCleanerRunLocked.
			// This guarantees that first opened connection is to be closed.
			// Thus it is timeOffset + secondTimeOffset + 3 (+2 for Close while reusing conns and +1 for Conn).
			10*time.Millisecond + 100*time.Nanosecond + 3*time.Nanosecond,
			time.Nanosecond,
			// Closed all not reused connections and extra one by max lifetime.
			int64(usedConns - reusedConns + 1),
			int64(usedConns - reusedConns),
			10 * time.Millisecond,
			// Add second offset because otherwise connections are expired via max lifetime in Close.
			100 * time.Nanosecond,
		},
		{
			time.Hour,
			0,
			time.Second,
			0,
			0,
			10 * time.Millisecond,
			0},
	}
	baseTime := time.Unix(0, 0)
	defer func() {
		nowFunc = time.Now
	}()
	for _, item := range list {
		nowFunc = func { baseTime }
		t.Run(fmt.Sprintf("%v", item.wantMaxIdleTime), func { t ->
			db := newTestDB(t, "people")
			defer closeDB(t, db)

			db.SetMaxOpenConns(usedConns)
			db.SetMaxIdleConns(usedConns)
			db.SetConnMaxIdleTime(item.wantMaxIdleTime)
			db.SetConnMaxLifetime(item.wantMaxLifetime)

			preMaxIdleClosed := db.Stats().MaxIdleTimeClosed

			// Busy usedConns.
			testUseConns(t, usedConns, baseTime, db)

			tm := baseTime.Add(item.timeOffset)

			// Reuse connections which should never be considered idle
			// and exercises the sorting for issue 39471.
			tm = testUseConns(t, reusedConns, tm, db)

			tm = tm.Add(item.secondTimeOffset)
			nowFunc = func { tm }

			db.mu.Lock()
			nc, closing := db.connectionCleanerRunLocked(time.Second)
			if nc != item.wantNextCheck {
				t.Errorf("got %v; want %v next check duration", nc, item.wantNextCheck)
			}

			// Validate freeConn order.
			var last time.Time
			for _, c := range db.freeConn {
				if last.After(c.returnedAt) {
					t.Error("freeConn is not ordered by returnedAt")
					break
				}
				last = c.returnedAt
			}

			db.mu.Unlock()
			for _, c := range closing {
				c.Close()
			}
			if g, w := int64(len(closing)), item.wantIdleClosed; g != w {
				t.Errorf("got: %d; want %d closed conns", g, w)
			}

			st := db.Stats()
			maxIdleClosed := st.MaxIdleTimeClosed - preMaxIdleClosed
			if g, w := maxIdleClosed, item.wantMaxIdleClosed; g != w {
				t.Errorf("got: %d; want %d max idle closed conns", g, w)
			}
		})
	}
}

type nvcDriver struct {
	fakeDriver
	skipNamedValueCheck bool
}

func (d *nvcDriver) Open(dsn string) (driver.Conn, error) {
	c, err := d.fakeDriver.Open(dsn)
	fc := c.(*fakeConn)
	fc.db.allowAny = true
	return &nvcConn{fc, d.skipNamedValueCheck}, err
}

type nvcConn struct {
	*fakeConn
	skipNamedValueCheck bool
}

type decimalInt struct {
	value int
}

type doNotInclude struct{}

var _ driver.NamedValueChecker = &nvcConn{}

func (c *nvcConn) CheckNamedValue(nv *driver.NamedValue) error {
	if c.skipNamedValueCheck {
		return driver.ErrSkip
	}
	switch v := nv.Value.(type) {
	default:
		return driver.ErrSkip
	case Out:
		switch ov := v.Dest.(type) {
		default:
			return errors.New("unknown NameValueCheck OUTPUT type")
		case *string:
			*ov = "from-server"
			nv.Value = "OUT:*string"
		}
		return nil
	case decimalInt, []int64:
		return nil
	case doNotInclude:
		return driver.ErrRemoveArgument
	}
}

func TestNamedValueChecker(t *testing.T) {
	Register("NamedValueCheck", &nvcDriver{})
	db, err := Open("NamedValueCheck", "")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_, err = db.ExecContext(ctx, "WIPE")
	if err != nil {
		t.Fatal("exec wipe", err)
	}

	_, err = db.ExecContext(ctx, "CREATE|keys|dec1=any,str1=string,out1=string,array1=any")
	if err != nil {
		t.Fatal("exec create", err)
	}

	o1 := ""
	_, err = db.ExecContext(ctx, "INSERT|keys|dec1=?A,str1=?,out1=?O1,array1=?", Named("A", decimalInt{123}), "hello", Named("O1", Out{Dest: &o1}), []int64{42, 128, 707}, doNotInclude{})
	if err != nil {
		t.Fatal("exec insert", err)
	}
	var (
		str1 string
		dec1 decimalInt
		arr1 []int64
	)
	err = db.QueryRowContext(ctx, "SELECT|keys|dec1,str1,array1|").Scan(&dec1, &str1, &arr1)
	if err != nil {
		t.Fatal("select", err)
	}

	list := []struct{ got, want any }{
		{o1, "from-server"},
		{dec1, decimalInt{123}},
		{str1, "hello"},
		{arr1, []int64{42, 128, 707}},
	}

	for index, item := range list {
		if !reflect.DeepEqual(item.got, item.want) {
			t.Errorf("got %#v wanted %#v for index %d", item.got, item.want, index)
		}
	}
}

func TestNamedValueCheckerSkip(t *testing.T) {
	Register("NamedValueCheckSkip", &nvcDriver{skipNamedValueCheck: true})
	db, err := Open("NamedValueCheckSkip", "")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	_, err = db.ExecContext(ctx, "WIPE")
	if err != nil {
		t.Fatal("exec wipe", err)
	}

	_, err = db.ExecContext(ctx, "CREATE|keys|dec1=any")
	if err != nil {
		t.Fatal("exec create", err)
	}

	_, err = db.ExecContext(ctx, "INSERT|keys|dec1=?A", Named("A", decimalInt{123}))
	if err == nil {
		t.Fatalf("expected error with bad argument, got %v", err)
	}
}

func TestOpenConnector(t *testing.T) {
	Register("testctx", &fakeDriverCtx{})
	db, err := Open("testctx", "people")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	c, ok := db.connector.(*fakeConnector)
	if !ok {
		t.Fatal("not using *fakeConnector")
	}

	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	if !c.closed {
		t.Fatal("connector is not closed")
	}
}

type ctxOnlyDriver struct {
	fakeDriver
}

func (d *ctxOnlyDriver) Open(dsn string) (driver.Conn, error) {
	conn, err := d.fakeDriver.Open(dsn)
	if err != nil {
		return nil, err
	}
	return &ctxOnlyConn{fc: conn.(*fakeConn)}, nil
}

var (
	_ driver.Conn           = &ctxOnlyConn{}
	_ driver.QueryerContext = &ctxOnlyConn{}
	_ driver.ExecerContext  = &ctxOnlyConn{}
)

type ctxOnlyConn struct {
	fc *fakeConn

	queryCtxCalled bool
	execCtxCalled  bool
}

func (c *ctxOnlyConn) Begin() (driver.Tx, error) {
	return c.fc.Begin()
}

func (c *ctxOnlyConn) Close() error {
	return c.fc.Close()
}

// Prepare is still part of the Conn interface, so while it isn't used
// must be defined for compatibility.
func (c *ctxOnlyConn) Prepare(q string) (driver.Stmt, error) {
	panic("not used")
}

func (c *ctxOnlyConn) PrepareContext(ctx context.Context, q string) (driver.Stmt, error) {
	return c.fc.PrepareContext(ctx, q)
}

func (c *ctxOnlyConn) QueryContext(ctx context.Context, q string, args []driver.NamedValue) (driver.Rows, error) {
	c.queryCtxCalled = true
	return c.fc.QueryContext(ctx, q, args)
}

func (c *ctxOnlyConn) ExecContext(ctx context.Context, q string, args []driver.NamedValue) (driver.Result, error) {
	c.execCtxCalled = true
	return c.fc.ExecContext(ctx, q, args)
}

// TestQueryExecContextOnly ensures drivers only need to implement QueryContext
// and ExecContext methods.
func TestQueryExecContextOnly(t *testing.T) {
	// Ensure connection does not implement non-context interfaces.
	var connType driver.Conn = &ctxOnlyConn{}
	if _, ok := connType.(driver.Execer); ok {
		t.Fatalf("%T must not implement driver.Execer", connType)
	}
	if _, ok := connType.(driver.Queryer); ok {
		t.Fatalf("%T must not implement driver.Queryer", connType)
	}

	Register("ContextOnly", &ctxOnlyDriver{})
	db, err := Open("ContextOnly", "")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	conn, err := db.Conn(ctx)
	if err != nil {
		t.Fatal("db.Conn", err)
	}
	defer conn.Close()
	coc := conn.dc.ci.(*ctxOnlyConn)
	coc.fc.skipDirtySession = true

	_, err = conn.ExecContext(ctx, "WIPE")
	if err != nil {
		t.Fatal("exec wipe", err)
	}

	_, err = conn.ExecContext(ctx, "CREATE|keys|v1=string")
	if err != nil {
		t.Fatal("exec create", err)
	}
	expectedValue := "value1"
	_, err = conn.ExecContext(ctx, "INSERT|keys|v1=?", expectedValue)
	if err != nil {
		t.Fatal("exec insert", err)
	}
	rows, err := conn.QueryContext(ctx, "SELECT|keys|v1|")
	if err != nil {
		t.Fatal("query select", err)
	}
	v1 := ""
	for rows.Next() {
		err = rows.Scan(&v1)
		if err != nil {
			t.Fatal("rows scan", err)
		}
	}
	rows.Close()

	if v1 != expectedValue {
		t.Fatalf("expected %q, got %q", expectedValue, v1)
	}

	if !coc.execCtxCalled {
		t.Error("ExecContext not called")
	}
	if !coc.queryCtxCalled {
		t.Error("QueryContext not called")
	}
}

type alwaysErrScanner struct{}

var errTestScanWrap = errors.New("errTestScanWrap")

func (alwaysErrScanner) Scan(any) error {
	return errTestScanWrap
}

// Issue 38099: Ensure that Rows.Scan properly wraps underlying errors.
func TestRowsScanProperlyWrapsErrors(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	rows, err := db.Query("SELECT|people|age|")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}

	var res alwaysErrScanner

	for rows.Next() {
		err = rows.Scan(&res)
		if err == nil {
			t.Fatal("expecting back an error")
		}
		if !errors.Is(err, errTestScanWrap) {
			t.Fatalf("errors.Is mismatch\n%v\nWant: %v", err, errTestScanWrap)
		}
		// Ensure that error substring matching still correctly works.
		if !strings.Contains(err.Error(), errTestScanWrap.Error()) {
			t.Fatalf("Error %v does not contain %v", err, errTestScanWrap)
		}
	}
}

type alwaysErrValuer struct{}

// errEmpty is returned when an empty value is found
var errEmpty = errors.New("empty value")

func (v alwaysErrValuer) Value() (driver.Value, error) {
	return nil, errEmpty
}

// Issue 64707: Ensure that Stmt.Exec and Stmt.Query properly wraps underlying errors.
func TestDriverArgsWrapsErrors(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	t.Run("exec", func { t ->
		_, err := db.Exec("INSERT|keys|dec1=?", alwaysErrValuer{})
		if err == nil {
			t.Fatal("expecting back an error")
		}
		if !errors.Is(err, errEmpty) {
			t.Fatalf("errors.Is mismatch\n%v\nWant: %v", err, errEmpty)
		}
		// Ensure that error substring matching still correctly works.
		if !strings.Contains(err.Error(), errEmpty.Error()) {
			t.Fatalf("Error %v does not contain %v", err, errEmpty)
		}
	})

	t.Run("query", func { t ->
		_, err := db.Query("INSERT|keys|dec1=?", alwaysErrValuer{})
		if err == nil {
			t.Fatal("expecting back an error")
		}
		if !errors.Is(err, errEmpty) {
			t.Fatalf("errors.Is mismatch\n%v\nWant: %v", err, errEmpty)
		}
		// Ensure that error substring matching still correctly works.
		if !strings.Contains(err.Error(), errEmpty.Error()) {
			t.Fatalf("Error %v does not contain %v", err, errEmpty)
		}
	})
}

func TestContextCancelDuringRawBytesScan(t *testing.T) {
	for _, mode := range []string{"nocancel", "top", "bottom", "go"} {
		t.Run(mode, func { t -> testContextCancelDuringRawBytesScan(t, mode) })
	}
}

// From go.dev/issue/60304
func testContextCancelDuringRawBytesScan(t *testing.T, mode string) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	if _, err := db.Exec("USE_RAWBYTES"); err != nil {
		t.Fatal(err)
	}

	// cancel used to call close asynchronously.
	// This test checks that it waits so as not to interfere with RawBytes.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	r, err := db.QueryContext(ctx, "SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	numRows := 0
	var sink byte
	for r.Next() {
		if mode == "top" && numRows == 2 {
			// cancel between Next and Scan is observed by Scan as err = context.Canceled.
			// The sleep here is only to make it more likely that the cancel will be observed.
			// If not, the test should still pass, like in "go" mode.
			cancel()
			time.Sleep(100 * time.Millisecond)
		}
		numRows++
		var s RawBytes
		err = r.Scan(&s)
		if numRows == 3 && err == context.Canceled {
			if r.closemuScanHold {
				t.Errorf("expected closemu NOT to be held")
			}
			break
		}
		if !r.closemuScanHold {
			t.Errorf("expected closemu to be held")
		}
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("read %q", s)
		if mode == "bottom" && numRows == 2 {
			// cancel before Next should be observed by Next, exiting the loop.
			// The sleep here is only to make it more likely that the cancel will be observed.
			// If not, the test should still pass, like in "go" mode.
			cancel()
			time.Sleep(100 * time.Millisecond)
		}
		if mode == "go" && numRows == 2 {
			// cancel at any future time, to catch other cases
			go cancel()
		}
		for _, b := range s { // some operation reading from the raw memory
			sink += b
		}
	}
	if r.closemuScanHold {
		t.Errorf("closemu held; should not be")
	}

	// There are 3 rows. We canceled after reading 2 so we expect either
	// 2 or 3 depending on how the awaitDone goroutine schedules.
	switch numRows {
	case 0, 1:
		t.Errorf("got %d rows; want 2+", numRows)
	case 2:
		if err := r.Err(); err != context.Canceled {
			t.Errorf("unexpected error: %v (%T)", err, err)
		}
	default:
		// Made it to the end. This is rare, but fine. Permit it.
	}

	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestContextCancelBetweenNextAndErr(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	r, err := db.QueryContext(ctx, "SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	for r.Next() {
	}
	cancel()                          // wake up the awaitDone goroutine
	time.Sleep(10 * time.Millisecond) // increase odds of seeing failure
	if err := r.Err(); err != nil {
		t.Fatal(err)
	}
}

func TestNilErrorAfterClose(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	// This WithCancel is important; Rows contains an optimization to avoid
	// spawning a goroutine when the query/transaction context cannot be
	// canceled, but this test tests a bug which is caused by said goroutine.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	r, err := db.QueryContext(ctx, "SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}

	if err := r.Close(); err != nil {
		t.Fatal(err)
	}

	time.Sleep(10 * time.Millisecond) // increase odds of seeing failure
	if err := r.Err(); err != nil {
		t.Fatal(err)
	}
}

// Issue #65201.
//
// If a RawBytes is reused across multiple queries,
// subsequent queries shouldn't overwrite driver-owned memory from previous queries.
func TestRawBytesReuse(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	if _, err := db.Exec("USE_RAWBYTES"); err != nil {
		t.Fatal(err)
	}

	var raw RawBytes

	// The RawBytes in this query aliases driver-owned memory.
	rows, err := db.Query("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	rows.Next()
	rows.Scan(&raw) // now raw is pointing to driver-owned memory
	name1 := string(raw)
	rows.Close()

	// The RawBytes in this query does not alias driver-owned memory.
	rows, err = db.Query("SELECT|people|age|")
	if err != nil {
		t.Fatal(err)
	}
	rows.Next()
	rows.Scan(&raw) // this must not write to the driver-owned memory in raw
	rows.Close()

	// Repeat the first query. Nothing should have changed.
	rows, err = db.Query("SELECT|people|name|")
	if err != nil {
		t.Fatal(err)
	}
	rows.Next()
	rows.Scan(&raw) // raw points to driver-owned memory again
	name2 := string(raw)
	rows.Close()
	if name1 != name2 {
		t.Fatalf("Scan read name %q, want %q", name2, name1)
	}
}

// badConn implements a bad driver.Conn, for TestBadDriver.
// The Exec method panics.
type badConn struct{}

func (bc badConn) Prepare(query string) (driver.Stmt, error) {
	return nil, errors.New("badConn Prepare")
}

func (bc badConn) Close() error {
	return nil
}

func (bc badConn) Begin() (driver.Tx, error) {
	return nil, errors.New("badConn Begin")
}

func (bc badConn) Exec(query string, args []driver.Value) (driver.Result, error) {
	panic("badConn.Exec")
}

// badDriver is a driver.Driver that uses badConn.
type badDriver struct{}

func (bd badDriver) Open(name string) (driver.Conn, error) {
	return badConn{}, nil
}

// Issue 15901.
func TestBadDriver(t *testing.T) {
	Register("bad", badDriver{})
	db, err := Open("bad", "ignored")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic")
		} else {
			if want := "badConn.Exec"; r.(string) != want {
				t.Errorf("panic was %v, expected %v", r, want)
			}
		}
	}()
	defer db.Close()
	db.Exec("ignored")
}

type pingDriver struct {
	fails bool
}

type pingConn struct {
	badConn
	driver *pingDriver
}

var pingError = errors.New("Ping failed")

func (pc pingConn) Ping(ctx context.Context) error {
	if pc.driver.fails {
		return pingError
	}
	return nil
}

var _ driver.Pinger = pingConn{}

func (pd *pingDriver) Open(name string) (driver.Conn, error) {
	return pingConn{driver: pd}, nil
}

func TestPing(t *testing.T) {
	driver := &pingDriver{}
	Register("ping", driver)

	db, err := Open("ping", "ignored")
	if err != nil {
		t.Fatal(err)
	}

	if err := db.Ping(); err != nil {
		t.Errorf("err was %#v, expected nil", err)
		return
	}

	driver.fails = true
	if err := db.Ping(); err != pingError {
		t.Errorf("err was %#v, expected pingError", err)
	}
}

// Issue 18101.
func TestTypedString(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)

	type Str string
	var scanned Str

	err := db.QueryRow("SELECT|people|name|name=?", "Alice").Scan(&scanned)
	if err != nil {
		t.Fatal(err)
	}
	expected := Str("Alice")
	if scanned != expected {
		t.Errorf("expected %+v, got %+v", expected, scanned)
	}
}

func BenchmarkConcurrentDBExec(b *testing.B) {
	b.ReportAllocs()
	ct := new(concurrentDBExecTest)
	for i := 0; i < b.N; i++ {
		doConcurrentTest(b, ct)
	}
}

func BenchmarkConcurrentStmtQuery(b *testing.B) {
	b.ReportAllocs()
	ct := new(concurrentStmtQueryTest)
	for i := 0; i < b.N; i++ {
		doConcurrentTest(b, ct)
	}
}

func BenchmarkConcurrentStmtExec(b *testing.B) {
	b.ReportAllocs()
	ct := new(concurrentStmtExecTest)
	for i := 0; i < b.N; i++ {
		doConcurrentTest(b, ct)
	}
}

func BenchmarkConcurrentTxQuery(b *testing.B) {
	b.ReportAllocs()
	ct := new(concurrentTxQueryTest)
	for i := 0; i < b.N; i++ {
		doConcurrentTest(b, ct)
	}
}

func BenchmarkConcurrentTxExec(b *testing.B) {
	b.ReportAllocs()
	ct := new(concurrentTxExecTest)
	for i := 0; i < b.N; i++ {
		doConcurrentTest(b, ct)
	}
}

func BenchmarkConcurrentTxStmtQuery(b *testing.B) {
	b.ReportAllocs()
	ct := new(concurrentTxStmtQueryTest)
	for i := 0; i < b.N; i++ {
		doConcurrentTest(b, ct)
	}
}

func BenchmarkConcurrentTxStmtExec(b *testing.B) {
	b.ReportAllocs()
	ct := new(concurrentTxStmtExecTest)
	for i := 0; i < b.N; i++ {
		doConcurrentTest(b, ct)
	}
}

func BenchmarkConcurrentRandom(b *testing.B) {
	b.ReportAllocs()
	ct := new(concurrentRandomTest)
	for i := 0; i < b.N; i++ {
		doConcurrentTest(b, ct)
	}
}

func BenchmarkManyConcurrentQueries(b *testing.B) {
	b.ReportAllocs()
	// To see lock contention in Go 1.4, 16~ cores and 128~ goroutines are required.
	const parallelism = 16

	db := newTestDB(b, "magicquery")
	defer closeDB(b, db)
	db.SetMaxIdleConns(runtime.GOMAXPROCS(0) * parallelism)

	stmt, err := db.Prepare("SELECT|magicquery|op|op=?,millis=?")
	if err != nil {
		b.Fatal(err)
	}
	defer stmt.Close()

	b.SetParallelism(parallelism)
	b.RunParallel(func { pb ->
		for pb.Next() {
			rows, err := stmt.Query("sleep", 1)
			if err != nil {
				b.Error(err)
				return
			}
			rows.Close()
		}
	})
}

func TestGrabConnAllocs(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	if race.Enabled {
		t.Skip("skipping allocation test when using race detector")
	}
	c := new(Conn)
	ctx := context.Background()
	n := int(testing.AllocsPerRun(1000, func() {
		_, release, err := c.grabConn(ctx)
		if err != nil {
			t.Fatal(err)
		}
		release(nil)
	}))
	if n > 0 {
		t.Fatalf("Conn.grabConn allocated %v objects; want 0", n)
	}
}

func BenchmarkGrabConn(b *testing.B) {
	b.ReportAllocs()
	c := new(Conn)
	ctx := context.Background()
	for i := 0; i < b.N; i++ {
		_, release, err := c.grabConn(ctx)
		if err != nil {
			b.Fatal(err)
		}
		release(nil)
	}
}

func TestConnRequestSet(t *testing.T) {
	var s connRequestSet
	wantLen := func(want int) {
		t.Helper()
		if got := s.Len(); got != want {
			t.Errorf("Len = %d; want %d", got, want)
		}
		if want == 0 && !t.Failed() {
			if _, ok := s.TakeRandom(); ok {
				t.Fatalf("TakeRandom returned result when empty")
			}
		}
	}
	reset := func() { s = connRequestSet{} }

	t.Run("add-delete", func { t ->
		reset()
		wantLen(0)
		dh := s.Add(nil)
		wantLen(1)
		if !s.Delete(dh) {
			t.Fatal("failed to delete")
		}
		wantLen(0)
		if s.Delete(dh) {
			t.Error("delete worked twice")
		}
		wantLen(0)
	})
	t.Run("take-before-delete", func { t ->
		reset()
		ch1 := make(chan connRequest)
		dh := s.Add(ch1)
		wantLen(1)
		if got, ok := s.TakeRandom(); !ok || got != ch1 {
			t.Fatalf("wrong take; ok=%v", ok)
		}
		wantLen(0)
		if s.Delete(dh) {
			t.Error("unexpected delete after take")
		}
	})
	t.Run("get-take-many", func { t ->
		reset()
		m := map[chan connRequest]bool{}
		const N = 100
		var inOrder, backOut []chan connRequest
		for range N {
			c := make(chan connRequest)
			m[c] = true
			s.Add(c)
			inOrder = append(inOrder, c)
		}
		if s.Len() != N {
			t.Fatalf("Len = %v; want %v", s.Len(), N)
		}
		for s.Len() > 0 {
			c, ok := s.TakeRandom()
			if !ok {
				t.Fatal("failed to take when non-empty")
			}
			if !m[c] {
				t.Fatal("returned item not in remaining set")
			}
			delete(m, c)
			backOut = append(backOut, c)
		}
		if len(m) > 0 {
			t.Error("items remain in expected map")
		}
		if slices.Equal(inOrder, backOut) { // N! chance of flaking; N=100 is fine
			t.Error("wasn't random")
		}
	})
	t.Run("close-delete", func { t ->
		reset()
		ch := make(chan connRequest)
		dh := s.Add(ch)
		wantLen(1)
		s.CloseAndRemoveAll()
		wantLen(0)
		if s.Delete(dh) {
			t.Error("unexpected delete after CloseAndRemoveAll")
		}
	})
}

func BenchmarkConnRequestSet(b *testing.B) {
	var s connRequestSet
	for range b.N {
		for range 16 {
			s.Add(nil)
		}
		for range 8 {
			if _, ok := s.TakeRandom(); !ok {
				b.Fatal("want ok")
			}
		}
		for range 8 {
			s.Add(nil)
		}
		for range 16 {
			if _, ok := s.TakeRandom(); !ok {
				b.Fatal("want ok")
			}
		}
		if _, ok := s.TakeRandom(); ok {
			b.Fatal("unexpected ok")
		}
	}
}
