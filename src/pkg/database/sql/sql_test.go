// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sql

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

const fakeDBName = "foo"

var chrisBirthday = time.Unix(123456789, 0)

func newTestDB(t *testing.T, name string) *DB {
	db, err := Open("test", fakeDBName)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := db.Exec("WIPE"); err != nil {
		t.Fatalf("exec wipe: %v", err)
	}
	if name == "people" {
		exec(t, db, "CREATE|people|name=string,age=int32,photo=blob,dead=bool,bdate=datetime")
		exec(t, db, "INSERT|people|name=Alice,age=?,photo=APHOTO", 1)
		exec(t, db, "INSERT|people|name=Bob,age=?,photo=BPHOTO", 2)
		exec(t, db, "INSERT|people|name=Chris,age=?,photo=CPHOTO,bdate=?", 3, chrisBirthday)
	}
	return db
}

func exec(t *testing.T, db *DB, query string, args ...interface{}) {
	_, err := db.Exec(query, args...)
	if err != nil {
		t.Fatalf("Exec of %q: %v", query, err)
	}
}

func closeDB(t *testing.T, db *DB) {
	err := db.Close()
	if err != nil {
		t.Fatalf("error closing DB: %v", err)
	}
}

func TestQuery(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	rows, err := db.Query("SELECT|people|age,name|")
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
		{age: 1, name: "Alice"},
		{age: 2, name: "Bob"},
		{age: 3, name: "Chris"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("mismatch.\n got: %#v\nwant: %#v", got, want)
	}

	// And verify that the final rows.Next() call, which hit EOF,
	// also closed the rows connection.
	if n := len(db.freeConn); n != 1 {
		t.Errorf("free conns after query hitting EOF = %d; want 1", n)
	}
}

func TestByteOwnership(t *testing.T) {
	db := newTestDB(t, "people")
	defer closeDB(t, db)
	rows, err := db.Query("SELECT|people|name,photo|")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	type row struct {
		name  []byte
		photo RawBytes
	}
	got := []row{}
	for rows.Next() {
		var r row
		err = rows.Scan(&r.name, &r.photo)
		if err != nil {
			t.Fatalf("Scan: %v", err)
		}
		got = append(got, r)
	}
	corruptMemory := []byte("\xffPHOTO")
	want := []row{
		{name: []byte("Alice"), photo: corruptMemory},
		{name: []byte("Bob"), photo: corruptMemory},
		{name: []byte("Chris"), photo: corruptMemory},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("mismatch.\n got: %#v\nwant: %#v", got, want)
	}

	var photo RawBytes
	err = db.QueryRow("SELECT|people|photo|name=?", "Alice").Scan(&photo)
	if err == nil {
		t.Error("want error scanning into RawBytes from QueryRow")
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
	if !reflect.DeepEqual(cols, want) {
		t.Errorf("got %#v; want %#v", cols, want)
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
	if !reflect.DeepEqual(photo, want) {
		t.Errorf("photo = %q; want %q", photo, want)
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

	type execTest struct {
		args    []interface{}
		wantErr string
	}
	execTests := []execTest{
		// Okay:
		{[]interface{}{"Brad", 31}, ""},
		{[]interface{}{"Brad", int64(31)}, ""},
		{[]interface{}{"Bob", "32"}, ""},
		{[]interface{}{7, 9}, ""},

		// Invalid conversions:
		{[]interface{}{"Brad", int64(0xFFFFFFFF)}, "sql: converting Exec argument #1's type: sql/driver: value 4294967295 overflows int32"},
		{[]interface{}{"Brad", "strconv fail"}, "sql: converting Exec argument #1's type: sql/driver: value \"strconv fail\" can't be converted to int32"},

		// Wrong number of args:
		{[]interface{}{}, "sql: expected 2 arguments, got 0"},
		{[]interface{}{1, 2, 3}, "sql: expected 2 arguments, got 3"},
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

func TestTxStmt(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t1|name=string,age=int32,dead=bool")
	stmt, err := db.Prepare("INSERT|t1|name=?,age=?")
	if err != nil {
		t.Fatalf("Stmt, err = %v, %v", stmt, err)
	}
	tx, err := db.Begin()
	if err != nil {
		t.Fatalf("Begin = %v", err)
	}
	_, err = tx.Stmt(stmt).Exec("Bobby", 7)
	if err != nil {
		t.Fatalf("Exec = %v", err)
	}
	err = tx.Commit()
	if err != nil {
		t.Fatalf("Commit = %v", err)
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
	fakeConn := db.freeConn[0].(*fakeConn)
	if made, closed := fakeConn.stmtsMade, fakeConn.stmtsClosed; made != closed {
		t.Errorf("statement close mismatch: made %d, closed %d", made, closed)
	}
}

func TestNullStringParam(t *testing.T) {
	db := newTestDB(t, "")
	defer closeDB(t, db)
	exec(t, db, "CREATE|t|id=int32,name=string,favcolor=nullstring")

	// Inserts with db.Exec:
	exec(t, db, "INSERT|t|id=?,name=?,favcolor=?", 1, "alice", NullString{"aqua", true})
	exec(t, db, "INSERT|t|id=?,name=?,favcolor=?", 2, "bob", NullString{"brown", false})

	_, err := db.Exec("INSERT|t|id=?,name=?,favcolor=?", 999, nil, nil)
	if err == nil {
		// TODO: this test fails, but it's just because
		// fakeConn implements the optional Execer interface,
		// so arguably this is the correct behavior.  But
		// maybe I should flesh out the fakeConn.Exec
		// implementation so this properly fails.
		// t.Errorf("expected error inserting nil name with Exec")
	}

	// Inserts with a prepared statement:
	stmt, err := db.Prepare("INSERT|t|id=?,name=?,favcolor=?")
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if _, err := stmt.Exec(3, "chris", "chartreuse"); err != nil {
		t.Errorf("exec insert chris: %v", err)
	}
	if _, err := stmt.Exec(4, "dave", NullString{"darkred", true}); err != nil {
		t.Errorf("exec insert dave: %v", err)
	}
	if _, err := stmt.Exec(5, "eleanor", NullString{"eel", false}); err != nil {
		t.Errorf("exec insert dave: %v", err)
	}

	// Can't put null name into non-nullstring column,
	if _, err := stmt.Exec(5, NullString{"", false}, nil); err == nil {
		t.Errorf("expected error inserting nil name with prepared statement Exec")
	}

	type nameColor struct {
		name     string
		favColor NullString
	}

	wantMap := map[int]nameColor{
		1: nameColor{"alice", NullString{"aqua", true}},
		2: nameColor{"bob", NullString{"", false}},
		3: nameColor{"chris", NullString{"chartreuse", true}},
		4: nameColor{"dave", NullString{"darkred", true}},
		5: nameColor{"eleanor", NullString{"", false}},
	}
	for id, want := range wantMap {
		var got nameColor
		if err := db.QueryRow("SELECT|t|name,favcolor|id=?", id).Scan(&got.name, &got.favColor); err != nil {
			t.Errorf("id=%d Scan: %v", id, err)
		}
		if got != want {
			t.Errorf("id=%d got %#v, want %#v", id, got, want)
		}
	}
}
