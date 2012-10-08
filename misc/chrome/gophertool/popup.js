function openURL(url) {
  chrome.tabs.create({ "url": url })
}

window.addEventListener("load", function () {
  console.log("hacking gopher pop-up loaded.");
  document.getElementById("inputbox").focus();
});

window.addEventListener("submit", function () {
  console.log("submitting form");
  var box = document.getElementById("inputbox");
  box.focus();

  var t = box.value;
  if (t == "") {
    return false;
  }

  var success = function(url) {
    console.log("matched " + t + " to: " + url)
    box.value = "";
    openURL(url);
    return false;  // cancel form submission
  };

  var url = urlForInput(t);
  if (url) {
    return success(url);
  }

  console.log("no match for text: " + t)
  return false;
});

window.addEventListener("click", function () {
  openURL("http://build.golang.org/");
});
