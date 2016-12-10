# Given the entry of a SVG file from the Wikipedia index, this returns a
# URL where the SVG file can be downloaded.

require 'nokogiri'
require 'open-uri'

file_name = ARGV[0].split(":").last.gsub(" ", "_")
file_page = "https://en.wikipedia.org/wiki/File:#{file_name}"

begin
	doc = Nokogiri::XML(open(file_page))
	file_url = doc.at_css("#file").children.first.attr("href")
	if file_url[0..1] == "//"
		file_url = "https:" + file_url
	end
	puts file_url
rescue
end
