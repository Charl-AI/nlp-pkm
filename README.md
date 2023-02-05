# nlp-pkm
open source models + personal notes = ???

This is a custom NLP project for personal knowledge management. I want to be able to write completely unstructured notes and snippets. I want to be able to retrieve this knowledge with semantic search. I want to automatically generate traverseable graphs.

My issue with PKM is that it does not scale well without meticulously structured notes. This does not work for me. I like writing snippets/notes/thoughts quickly when they pop into my head; I do not enjoy organising them and I want to spend as little time as possible away from what I was initially doing. I found Obsidian (using a flat structure and only backlinks for organisation) nice, but now I want to see if I can design a system with even less friction.

NLP has been getting good for a few years now, so this is a hot space. There seem to be quite a few startups working on similar solutions. Others are developing tools that wrap around OpenAI APIs. I want to build my own system based on the best open source models available. This is so I can customise it to my needs and avoid counterparty risk (i.e. company death / pricing structure changes / data leaks).


## Wishlist 🚀

- **Unstructured notes**. No schema or structure. File boundaries should be meaningless: notes in atomic files should work just as well as disparate concepts strung together in a large file.

- **Semantic search**. Retrieve notes based on ideas / areas. No need to remember keywords or file names.

- **Graphs / Spatial Relationships**. Surprisingly, I find knowledge retrieval easier in paper notebooks than digital ones. I this this is because I am able to associate an idea with a physical region (i.e. how far into the book it is). This sort of spatial reasoning works well for me (and probably many others). I want to be able to do this with digital notes. Notes should be associated with a *persistant* location in a graph, with similar notes being physically close. Absolute positioning of each note in the graph should remain consistent so I can learn to find things through muscle memory over time.

- **Conversational**. A nice-to-have would be the ability to chat to my notes a-la ChatGPT. A chatbot could work like a combined search engine / writing aid / ideation machine and could synthesise my notes together to answer what I really want to know. I'm not 100% sold on this idea yet, but would be interesting to try once good open source models are available.

- **Extensible with arbitrary corpuses**. It would be cool to have a mode toggle -- one mode only sees my notes -- the other sees my notes and ArXiv + Wikipedia + StackOverflow + etc. I could then see how my notes fit into current research trends using the graph, get ultra-personalised internet search results, and have a chatbot that sees how my work fits in with the wider world.

## TODO 🛠

- Figure out UX. Currently using python + argparse. Perhaps it could be a CLI application with Rust + Py03? Or should I make a vscode extension? I don't know anything about typescript at the moment -- could I use Rust + wasm?
