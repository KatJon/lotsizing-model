all:
	@echo "use specific commands"

.PHONY: fig
fig:
	zip fig.zip results/* results/or26/*

.PHONY: copyfigs
copyfigs:
	cp -f results/*.pdf ../lotsizing25/fig
	cp -f results/*.txt ../lotsizing25
