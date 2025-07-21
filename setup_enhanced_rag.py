#!/usr/bin/env python3
"""
Enhanced RAG Setup Script
Installs and configures dependencies for the enhanced RAG system
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"   Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_requirements():
    """Install requirements from requirements.txt"""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )

def install_spacy_model():
    """Install spaCy English model"""
    return run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Installing spaCy English model"
    )

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing imports...")
    
    packages = [
        ("crewai", "CrewAI"),
        ("chromadb", "ChromaDB"),
        ("sklearn", "scikit-learn"),
        ("numpy", "NumPy"),
        ("spacy", "spaCy"),
        ("docling", "Docling")
    ]
    
    all_good = True
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            all_good = False
    
    # Test spaCy model specifically
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy English model loaded successfully")
    except Exception as e:
        print(f"❌ spaCy English model failed to load: {e}")
        all_good = False
    
    return all_good

def check_env_file():
    """Check if .env file exists and has required variables"""
    print("\n🔍 Checking environment configuration...")
    
    if not os.path.exists(".env"):
        print("⚠️ .env file not found")
        print("💡 Create a .env file with your API keys:")
        print("   OPENAI_API_KEY=your_openai_key_here")
        print("   OPENROUTER_API_KEY=your_openrouter_key_here")
        return False
    
    # Read .env file
    with open(".env", "r") as f:
        env_content = f.read()
    
    required_keys = ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if key not in env_content or f"{key}=" not in env_content:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"⚠️ Missing environment variables: {', '.join(missing_keys)}")
        print("💡 Add them to your .env file")
        return False
    
    print("✅ Environment configuration looks good")
    return True

def main():
    """Main setup function"""
    print("🚀 Enhanced RAG Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Install spaCy model
    if not install_spacy_model():
        print("⚠️ spaCy model installation failed, but continuing...")
        print("💡 You can install it manually later with:")
        print("   python -m spacy download en_core_web_sm")
    
    # Test imports
    if not test_imports():
        print("⚠️ Some imports failed, but setup may still work")
    
    # Check environment
    check_env_file()
    
    print("\n" + "=" * 50)
    print("🎯 Enhanced RAG Setup Summary:")
    print("✅ Enhanced retrieval with hybrid search")
    print("✅ Semantic chunking for better context")
    print("✅ Advanced answer choice analysis")
    print("✅ Context synthesis from multiple sources")
    print("✅ TF-IDF sparse retrieval")
    print("✅ Intelligent query reformulation")
    
    print("\n💡 Next steps:")
    print("1. Make sure your .env file has the required API keys")
    print("2. Re-process your documents with the enhanced chunking:")
    print("   python chromadb_manager.py")
    print("3. Run your Kahoot bot with improved accuracy:")
    print("   python kahoot_bot.py")
    
    print("\n🔍 If you experience issues:")
    print("- Check that all environment variables are set")
    print("- Ensure your ChromaDB collection has documents")
    print("- Try re-running this setup script")
    
    print("\n🎉 Enhanced RAG setup complete!")

if __name__ == "__main__":
    main() 