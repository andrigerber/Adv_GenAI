Dear Susie and Guang,

Attached is the standalone code; sorry for the delays.

This should run in a virtual Python environment, since langchain et al. are very version-dependent - things get deprecated approximately every five minutes. I am attaching a requirements file. reqs.txt.

Note that this is written for Azure AI Service. For OpenAI you might need to make some adjustments.

The server and key (as well as in the case of Azure AI the deployments) are in the file config_v2.txt - you would have to insert your own access data.

* digester_cmd_v2.py takes a directory from the command line and creates embeddings for all PDFs, output to a directory ./chroma_db
The path to Tesseract would be platform-dependent and needs to be adjusted.

* rag_retrieve_cmd_v2.py takes a prompt from the command line (in quotation marks)

* openaiclass_cmd_v2.py is a shared module

I hope this will be helpful for the students.

Please let me know if this arrived, I had some email problems lately with stuff ending up in spam at other universities.

Best,

- Gerd.
