# How this input was created

## Downloaded the index of the english wikipedia:

The file enwiki-20150602-pages-articles-multistream-index.txt.bz2 at this page: https://dumps.wikimedia.org/enwiki/20150602/

## Grepped out SVG files

    cat enwiki-20150602-pages-articles-multistream-index.txt.txt|grep .svg|grep File>svg_index.txt

## Converted index entries into URLs to download

See the script index_entry_to_svg_url.rb. It's extremely brittle, I just ignored errors and moved on.

### Example use

    ruby index_entry_to_svg_url.rb 12789746453:46843014:File:University\ of\ Warwick\ logo\ 2015\ with\ descriptor.svg
    https://upload.wikimedia.org/wikipedia/en/3/3e/University_of_Warwick_logo_2015_with_descriptor.svg

## curling each URL

In practice I did something like this

cat svg_index.txt|xargs -l ruby index_entry_to_svg_url.rb|xargs -l curl>svgdata.txt

## Cleaned up file

I used the following regext in my text editor to remove a bunch of CDATA tags.

    <!\[CDATA\[[^\]\]>]*]]> 

I also manually removed some PNG binaries which made it into the file.


