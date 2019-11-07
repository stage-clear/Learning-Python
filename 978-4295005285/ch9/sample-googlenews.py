from selenium import webdriver

base_url = 'https://news.google.com/news/?ned=us&l=en'

driver = webdriver.Chrome('../ch4/chromedriver')
driver.implicitly_wait(10)
driver.get(base_url)

js_cmd = '''
(function (d, script) {
    script = d.createElement('script')
    script.type = 'text/javascript'
    script.async = true;
    script.onload = function () {
        var documentClone = document.cloneNode(true)
        var loc = document.location;
        var uri = {
            spec: loc.href,
            host: loc.host,
            prePath: loc.protocol + '//' + loc.host,
            scheme: loc.protocol.substr(0, loc.protocol.indexOf(':')),
            pathBase: loc.protocol + '//' + loc.host +
                      loc.pathname.substr(0, loc.pathname.lastIndexOf('/') + 1)
        }
        var article = new Readability(uri, documentClone).parse()
        document.body.innerHTML = '<h1 id="title">' + article.title + '</h1>' +
            '<div id="content">' + article.content + '</div>';
    }
    script.src = 'path/to/Readability.js'
    d.getElementByTagName('head')[0].appendChild(script)
}(document));
'''

driver.execute_script(js_cmd)

title = driver.find_element_by_id('title').text.strip()
content = driver.find_element_by_id('content').text.strip()

print('Title was: ', title)

driver.quit()
