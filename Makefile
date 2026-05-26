all:
	@echo "use specific commands"

.PHONY: fig
fig:
	zip fig.zip results/*

.PHONY: copyfigs
copyfigs:
	cp -f results/*.pdf ../lotsizing25/fig
	cp -f results/*.txt ../lotsizing25