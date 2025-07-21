#!/usr/bin/env python3
"""
Enhanced CrewAI Flow-based Kahoot Bot using Browser MCP
A persistent agent that plays Kahoot quizzes automatically with improved RAG accuracy
"""

import os
import time
from typing import Dict
from pydantic import BaseModel
from crewai import Agent, LLM, Task
from crewai.flow.flow import Flow, listen, start
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from dotenv import load_dotenv
from rag_tool import EnhancedRAGTool  # Use the enhanced version

# Load environment variables
load_dotenv()

# Flow state model
class KahootBotState(BaseModel):
    game_pin: str = os.getenv("KAHOOT_PIN", "299268")
    nickname: str = os.getenv("KAHOOT_NICKNAME", "CrewAI_Bot")
    is_connected: bool = False

class KahootBotFlow(Flow[KahootBotState]):
    """Persistent Kahoot bot using CrewAI Flow and Browser MCP with Enhanced RAG"""
    
    def __init__(self):
        super().__init__()
        
        # Configure LLM
        self.llm = LLM(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Browser MCP server parameters - use the actual @browsermcp/mcp package
        env = os.environ.copy()
        env["SKIP_PORT_KILL"] = "1"
        self.browser_server_params = StdioServerParameters(
            command="npx",
            args=["@browsermcp/mcp@latest"],
            env=env
        )
        
    @start()
    def initialize_bot(self) -> KahootBotState:
        """Initialize the Kahoot bot"""
        print("🤖 Starting Enhanced Kahoot Bot with Improved RAG...")
        
        # Debug environment variables
        print(f"🐛 Debug - KAHOOT_PIN env var: '{os.getenv('KAHOOT_PIN')}'")
        print(f"🐛 Debug - KAHOOT_NICKNAME env var: '{os.getenv('KAHOOT_NICKNAME')}'")
        
        state = KahootBotState()
        print(f"🔧 Using PIN: {state.game_pin}")
        print(f"🔧 Using Nickname: {state.nickname}")
        
        return state
    
    @listen(initialize_bot)
    def play_quiz_actively(self, state: KahootBotState) -> KahootBotState:
        """Keep bot alive and answer questions one by one with enhanced RAG accuracy"""
        print(f"🎮 Starting enhanced quiz bot for {state.nickname}")
        print("📋 ASSUMPTION: You are already in the Kahoot waiting room")
        print("🚀 ENHANCEMENT: Using hybrid retrieval + semantic chunking for better accuracy!")

        # Prompt user to select a collection at the start of the session
        from chromadb import PersistentClient

        chroma_client = PersistentClient(path="./chroma_db")
        collections = chroma_client.list_collections()
        if not collections:
            print("❌ No collections found in ChromaDB. Please create one using chromadb_manager.py first.")
            return state

        print("📚 Available ChromaDB collections:")
        for idx, col in enumerate(collections):
            print(f"{idx+1}. {col.name}")

        selected_collection_name = None
        while selected_collection_name is None:
            try:
                choice = int(input("Select a collection by number for Enhanced RAGTool: ")) - 1
                if 0 <= choice < len(collections):
                    selected_collection_name = collections[choice].name
                    print(f"✅ Selected collection for Enhanced RAGTool: {selected_collection_name}")
                else:
                    print("❌ Invalid selection. Try again.")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")

        try:
            rag_tool = EnhancedRAGTool(collection_name=selected_collection_name, db_path="./chroma_db/chroma.sqlite3")
            print("🎯 Enhanced RAG Tool initialized with:")
            print("   ✅ Hybrid retrieval (semantic + keyword)")
            print("   ✅ Semantic chunking for better context")
            print("   ✅ Advanced answer choice analysis")
            print("   ✅ Context synthesis from multiple sources")
        except Exception as e:
            print(f"❌ Failed to initialize Enhanced RAGTool: {e}")
            rag_tool = None
        if not rag_tool:
            print("❌ Enhanced RAGTool initialization Failed.")
            return state

        try:
            with MCPServerAdapter(self.browser_server_params) as browser_tools:
                print("🚀 Browser MCP server started!")
                print("➡️  In Chrome, click the Browser-MCP extension icon, then 'Connect'.")
                print("   After it shows the green badge, come back here and press Enter...")
                
                # Wait for user to connect extension
                input("🔄  Press <Enter> when the extension is connected...")
                
                # Test connection
                while True:
                    try:
                        result = browser_tools.browser_snapshot(random_string="ping")
                        print(f"✅ Connection test successful: {len(str(result))} chars returned")
                        break
                    except Exception as e:
                        print(f"❌ Connection test failed: {str(e)}")
                        try:
                            tool_names = [tool.name for tool in browser_tools]
                            print(f"ℹ️  Tools available: {tool_names}")
                            if 'browser_navigate' in tool_names:
                                print("✅ Browser tools detected, proceeding anyway...")
                                break
                        except Exception as e2:
                            print(f"❌ Tools check also failed: {str(e2)}")
                        
                        print("Make sure the Browser MCP extension is connected (green badge) and press Enter...")
                        input()

                print("✅ Browser MCP connected!")
                print(f"Available tools: {[tool.name for tool in browser_tools]}")
                
                # Intelligent hybrid instructions with accuracy guarantee
                answer_instructions = f"""
                🧠 INTELLIGENT HYBRID WORKFLOW FOR SPEED + 100% ACCURACY:

                STEP 1: Take browser_snapshot to see the current page (REQUIRED)
                
                STEP 2: Extract BOTH from the browser snapshot:
                - The EXACT question text word-for-word
                - ALL available answer choices with their button references
                  Example: "People and Possibilities in the Age of AI" [ref=s1e43]
                  CRITICAL: Note the exact ref attribute for each button!
                
                STEP 3: Use INTELLIGENT HYBRID RAG for guaranteed accuracy:
                - Query: exact question text
                - answer_choices: ["choice1", "choice2", "choice3", "choice4"] 
                - The system automatically uses:
                  * FAST MODE first (2-3 seconds) for high-confidence answers
                  * FULL MODE automatically (5-7 seconds) if confidence is low
                - Look for confidence scores and validation details
                - TRUST recommendations with confidence ≥ 8.0 (VERY HIGH)
                - Consider recommendations with confidence ≥ 5.0 (HIGH)
                
                STEP 4: Click using CONFIDENCE-BASED STRATEGY:
                
                🎯 HIGH CONFIDENCE (≥8.0) - CLICK IMMEDIATELY:
                - Use exact ref: browser_click(ref="s1e44")
                - These are nearly guaranteed correct
                
                📈 MEDIUM-HIGH CONFIDENCE (5.0-7.9) - CLICK WITH VALIDATION:
                - Verify text match exactly before clicking
                - Use ref method: browser_click(ref="s1e44")
                
                📊 LOWER CONFIDENCE (<5.0) - USE BACKUP METHODS:
                - Try multiple clicking approaches
                - Verify in multiple evidence sources
                - Still click best option but with more verification

                🧠 INTELLIGENT FEATURES NOW ACTIVE:
                - Automatic fast→full escalation for accuracy
                - Cross-validation of answers across multiple sources
                - Context window analysis for semantic consistency
                - Evidence strength validation
                - Confidence scoring with validation breakdown

                ⚡ SPEED + ACCURACY GUARANTEES:
                - High-confidence answers: 2-3 seconds (fast mode)
                - Complex questions: 5-7 seconds (full mode + validation)
                - Cache hits: <1 second (instant)
                - Accuracy: Optimized for correctness with speed

                🎯 IMPROVED BUTTON MAPPING WITH CONFIDENCE:
                1. Hybrid RAG says: "🎯 HIGHLY RECOMMENDED ANSWER: 'brain structure changes'" (confidence: 8.5)
                2. In browser snapshot, find: button "brain structure changes" [ref=s1e44]
                3. Execute IMMEDIATELY (high confidence): browser_click(ref="s1e44")
                4. Success expected due to high validation scores

                🔧 CONFIDENCE-BASED CLICKING STRATEGY:
                - Confidence ≥8.0: Click immediately with ref method
                - Confidence 5.0-7.9: Verify text match, then click ref method
                - Confidence 3.0-4.9: Try ref, then backup methods if needed
                - Confidence <3.0: Multiple verification steps before clicking

                🚨 ACCURACY-FIRST RULES:
                - ✅ Trust high-confidence recommendations (≥8.0)
                - ✅ Use validation scores to assess answer quality
                - ✅ Check evidence strength and context consistency
                - ✅ Click immediately for validated high-confidence answers
                - ❌ Don't doubt validated high-confidence recommendations
                - ❌ Don't skip verification for medium confidence answers
                - ❌ Don't rush if confidence is low - use full validation

                INTELLIGENT CLICKING CHECKLIST:
                □ Checked confidence score and validation details
                □ Used appropriate clicking strategy for confidence level
                □ Found exact text match in browser snapshot
                □ Using correct ref ID for the validated answer

                Remember: This system balances speed and accuracy automatically!
                """

                # Continuous quiz loop
                question_count = 0
                print("\n🎯 Starting enhanced quiz mode with improved accuracy!")
                print("💡 Bot will answer one question, then wait for your input to continue")
                print("❌ Press Ctrl+C to stop the bot\n")
                
                while True:
                    try:
                        question_count += 1
                        print(f"🔍 Looking for question #{question_count} (Enhanced RAG active)...")
                        
                        # Create fresh agent for each question with Enhanced RAG
                        tools_list = list(browser_tools)
                        if rag_tool:
                            tools_list.append(rag_tool)

                        quiz_agent = Agent(
                            role="Intelligent Hybrid Kahoot Expert",
                            goal="ACCURACY + SPEED: Use intelligent hybrid RAG processing to get validated answers, then click with confidence-appropriate strategy.",
                            backstory="I am an intelligent quiz expert powered by Hybrid RAG that automatically balances speed and accuracy. I understand confidence scores and validation metrics. For high-confidence answers (≥8.0), I click immediately. For medium confidence (5.0-7.9), I verify carefully. For low confidence (<5.0), I use comprehensive validation. I never sacrifice accuracy for speed, but I'm smart about when to be fast vs thorough. The hybrid system ensures I get the best of both worlds.",
                            llm=self.llm,
                            tools=tools_list,
                            verbose=True,
                            memory=False,  # No memory to avoid cached browser state
                            max_iter=6,  # Increased for comprehensive validation when needed
                            allow_delegation=False,  # Prevent delegation issues
                        )
                        
                        quiz_task = Task(
                            description="INTELLIGENT PROCESSING: 1. Take browser snapshot 2. Extract question + answer choices with refs 3. Use Intelligent Hybrid RAG (auto fast→full escalation) 4. Apply confidence-based clicking strategy 5. ENSURE ACCURATE CLICK",
                            expected_output="Successfully clicked the correct answer button using confidence-appropriate method (immediate for high confidence, validated for medium/low), with detailed processing and validation information",
                            agent=quiz_agent,
                            tools=tools_list,
                            prompt=answer_instructions,
                        )

                        result = quiz_task.execute_sync()
                        print(f"✅ Question #{question_count} Result: {result}")
                        
                        # Wait for user to press Enter before continuing to next question
                        print(f"\n⏳ Question #{question_count} completed with Enhanced RAG!")
                        print("📋 Navigate to the next question in Kahoot (if any)")
                        next_action = input("🔄 Press <Enter> to look for next question, or type 'quit' to stop: ").strip().lower()
                        
                        if next_action == 'quit':
                            print("🛑 Stopping enhanced quiz bot as requested")
                            break
                            
                        print(f"\n{'='*50}")
                        
                    except KeyboardInterrupt:
                        print("\n🛑 Enhanced quiz bot stopped by user (Ctrl+C)")
                        break
                    except Exception as e:
                        print(f"❌ Error in question #{question_count}: {e}")
                        retry = input("🔄 Press <Enter> to retry, or type 'quit' to stop: ").strip().lower()
                        if retry == 'quit':
                            break

        except Exception as e:
            print(f"❌ Quiz monitoring failed: {e}")
            
        return state

def main():
    """Main entry point"""
    print("🚀 Enhanced CrewAI Kahoot Bot Starting...")
    print("🎯 Features: Hybrid RAG + Semantic Chunking + Advanced Answer Analysis")
    
    # Create and run the flow
    bot_flow = KahootBotFlow()
    result = bot_flow.kickoff()
    
    print(f"\n🎯 Final Result: {result}")

if __name__ == "__main__":
    main()
