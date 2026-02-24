import sys
import os

print(f"Python: {sys.executable}")
try:
    import langchain
    print(f"LangChain File: {langchain.__file__}")
    print(f"LangChain Version: {getattr(langchain, '__version__', 'unknown')}")
    print(f"LangChain Dir: {dir(langchain)}")
    
    try:
        import langchain.chains
        print(f"LangChain Chains File: {langchain.chains.__file__}")
        print(f"LangChain Chains Dir: {dir(langchain.chains)}")
    except ImportError as e:
        print(f"Error importing langchain.chains: {e}")
        
except ImportError as e:
    print(f"Error importing langchain: {e}")
