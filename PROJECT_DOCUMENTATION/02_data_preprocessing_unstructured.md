# Data Preprocessing (Unstructured) - Function Reference

## Core Functions

### `json.load()`
- **Purpose**: Parse JSON data from a file
- **Parameters**:
  - `fp`: File pointer object
- **Returns**: Python object (dict/list)

### `langchain.schema.Document`
- **Purpose**: Represents a document in the LangChain ecosystem
- **Key Attributes**:
  - `page_content`: The actual text content
  - `metadata`: Additional document metadata

### `RecursiveCharacterTextSplitter`
- **Purpose**: Split text into chunks
- **Key Parameters**:
  - `chunk_size`: Maximum size of chunks
  - `chunk_overlap`: Overlap between chunks
- **Methods**:
  - `split_documents()`: Split documents into chunks

### `pickle.dump()`
- **Purpose**: Serialize Python object to file
- **Parameters**:
  - `obj`: Object to serialize
  - `file`: File object opened in binary mode

## Data Structures

### `list`
- **Purpose**: Ordered collection of items
- **Use Case**: Store documents, chunks, etc.

### `dict`
- **Purpose**: Key-value storage
- **Use Case**: Store document metadata, configurations

## File Formats

### JSON (.json)
- **Structure**: List of Q&A pairs
- **Example**:
  ```json
  [
    {
      "question": "What is diabetes?",
      "answer": "Diabetes is a chronic condition..."
    }
  ]