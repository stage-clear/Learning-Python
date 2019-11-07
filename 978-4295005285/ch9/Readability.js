(function (d, script) {
  script = d.createElement('script')
  script.type = 'text/javascript'
  script.async = true
  script.onload = function () {
    console.log('The script was successfully injected!')
  }
  script.src = '/path/to/Readability.js'
}(document));
