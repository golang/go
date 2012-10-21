function openURL(url) {
  chrome.tabs.create({ "url": url })
}

function addLinks() {
  var links = document.getElementsByTagName("a");
  for (var i = 0; i < links.length; i++) {
    var url = links[i].getAttribute("url");
    if (url)
      links[i].addEventListener("click", function () {
        openURL(this.getAttribute("url"));
      });
  }
}

window.addEventListener("load", function () {
  addLinks();
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
