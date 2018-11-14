package main

import (
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

var AuthorPaidByTheColumnInch struct {
	fog int `
	London. Michaelmas term lately over, and the Lord Chancellor sitting in Lincoln’s Inn Hall. Implacable November weather. As much mud in the streets as if the waters had but newly retired from the face of the earth, and it would not be wonderful to meet a Megalosaurus, forty feet long or so, waddling like an elephantine lizard up Holborn Hill. Smoke lowering down from chimney-pots, making a soft black drizzle, with flakes of soot in it as big as full-grown snowflakes—gone into mourning, one might imagine, for the death of the sun. Dogs, undistinguishable in mire. Horses, scarcely better; splashed to their very blinkers. Foot passengers, jostling one another’s umbrellas in a general infection of ill temper, and losing their foot-hold at street-corners, where tens of thousands of other foot passengers have been slipping and sliding since the day broke (if this day ever broke), adding new deposits to the crust upon crust of mud, sticking at those points tenaciously to the pavement, and accumulating at compound interest.

	Fog everywhere. Fog up the river, where it flows among green aits and meadows; fog down the river, where it rolls defiled among the tiers of shipping and the waterside pollutions of a great (and dirty) city. Fog on the Essex marshes, fog on the Kentish heights. Fog creeping into the cabooses of collier-brigs; fog lying out on the yards and hovering in the rigging of great ships; fog drooping on the gunwales of barges and small boats. Fog in the eyes and throats of ancient Greenwich pensioners, wheezing by the firesides of their wards; fog in the stem and bowl of the afternoon pipe of the wrathful skipper, down in his close cabin; fog cruelly pinching the toes and fingers of his shivering little ‘prentice boy on deck. Chance people on the bridges peeping over the parapets into a nether sky of fog, with fog all round them, as if they were up in a balloon and hanging in the misty clouds.

	Gas looming through the fog in divers places in the streets, much as the sun may, from the spongey fields, be seen to loom by husbandman and ploughboy. Most of the shops lighted two hours before their time—as the gas seems to know, for it has a haggard and unwilling look.

	The raw afternoon is rawest, and the dense fog is densest, and the muddy streets are muddiest near that leaden-headed old obstruction, appropriate ornament for the threshold of a leaden-headed old corporation, Temple Bar. And hard by Temple Bar, in Lincoln’s Inn Hall, at the very heart of the fog, sits the Lord High Chancellor in his High Court of Chancery.`

	wind int `
	It was grand to see how the wind awoke, and bent the trees, and drove the rain before it like a cloud of smoke; and to hear the solemn thunder, and to see the lightning; and while thinking with awe of the tremendous powers by which our little lives are encompassed, to consider how beneficent they are, and how upon the smallest flower and leaf there was already a freshness poured from all this seeming rage, which seemed to make creation new again.`

	jarndyce int `
	Jarndyce and Jarndyce drones on. This scarecrow of a suit has, over the course of time, become so complicated, that no man alive knows what it means. The parties to it understand it least; but it has been observed that no two Chancery lawyers can talk about it for five minutes, without coming to a total disagreement as to all the premises. Innumerable children have been born into the cause; innumerable young people have married into it; innumerable old people have died out of it. Scores of persons have deliriously found themselves made parties in Jarndyce and Jarndyce, without knowing how or why; whole families have inherited legendary hatreds with the suit. The little plaintiff or defendant, who was promised a new rocking-horse when Jarndyce and Jarndyce should be settled, has grown up, possessed himself of a real horse, and trotted away into the other world. Fair wards of court have faded into mothers and grandmothers; a long procession of Chancellors has come in and gone out; the legion of bills in the suit have been transformed into mere bills of mortality; there are not three Jarndyces left upon the earth perhaps, since old Tom Jarndyce in despair blew his brains out at a coffee-house in Chancery Lane; but Jarndyce and Jarndyce still drags its dreary length before the Court, perennially hopeless.`

	principle int `
	The one great principle of the English law is, to make business for itself. There is no other principle distinctly, certainly, and consistently maintained through all its narrow turnings. Viewed by this light it becomes a coherent scheme, and not the monstrous maze the laity are apt to think it. Let them but once clearly perceive that its grand principle is to make business for itself at their expense, and surely they will cease to grumble.`
}

func TestLargeSymName(t *testing.T) {
	// The compiler generates a symbol name using the string form of the
	// type. This tests that the linker can read symbol names larger than
	// the bufio buffer. Issue #15104.
	_ = AuthorPaidByTheColumnInch
}

func TestIssue21703(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	const source = `
package main
const X = "\n!\n"
func main() {}
`

	tmpdir, err := ioutil.TempDir("", "issue21703")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v\n", err)
	}
	defer os.RemoveAll(tmpdir)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "main.go"), []byte(source), 0666)
	if err != nil {
		t.Fatalf("failed to write main.go: %v\n", err)
	}

	cmd := exec.Command(testenv.GoToolPath(t), "tool", "compile", "main.go")
	cmd.Dir = tmpdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to compile main.go: %v, output: %s\n", err, out)
	}

	cmd = exec.Command(testenv.GoToolPath(t), "tool", "link", "main.o")
	cmd.Dir = tmpdir
	out, err = cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("failed to link main.o: %v, output: %s\n", err, out)
	}
}

// TestIssue28429 ensures that the linker does not attempt to link
// sections not named *.o. Such sections may be used by a build system
// to, for example, save facts produced by a modular static analysis
// such as golang.org/x/tools/go/analysis.
func TestIssue28429(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	tmpdir, err := ioutil.TempDir("", "issue28429-")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	write := func(name, content string) {
		err := ioutil.WriteFile(filepath.Join(tmpdir, name), []byte(content), 0666)
		if err != nil {
			t.Fatal(err)
		}
	}

	runGo := func(args ...string) {
		cmd := exec.Command(testenv.GoToolPath(t), args...)
		cmd.Dir = tmpdir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("'go %s' failed: %v, output: %s",
				strings.Join(args, " "), err, out)
		}
	}

	// Compile a main package.
	write("main.go", "package main; func main() {}")
	runGo("tool", "compile", "-p", "main", "main.go")
	runGo("tool", "pack", "c", "main.a", "main.o")

	// Add an extra section with a short, non-.o name.
	// This simulates an alternative build system.
	write(".facts", "this is not an object file")
	runGo("tool", "pack", "r", "main.a", ".facts")

	// Verify that the linker does not attempt
	// to compile the extra section.
	runGo("tool", "link", "main.a")
}
