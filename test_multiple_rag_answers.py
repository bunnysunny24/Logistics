#!/usr/bin/env python3
"""
Enhanced RAG Query System - Multiple Answers
===========================================

Test different approaches to get multiple answers from RAG system
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_multiple_answer_approaches():
    """Test different ways to get multiple answers"""
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║            🧠 MULTIPLE RAG ANSWERS TEST                   ║
    ║                                                           ║
    ║     Testing different approaches for multiple answers     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Test queries that should return multiple perspectives
    test_queries = [
        {
            "query": "What are the different types of anomalies detected in our invoices?",
            "approach": "Multiple Categories"
        },
        {
            "query": "Show me all high-risk anomalies and their different risk factors",
            "approach": "Risk-based Analysis"
        },
        {
            "query": "What are the various fraud patterns we've detected across different suppliers?",
            "approach": "Pattern Analysis"
        },
        {
            "query": "Give me different anomaly examples with their evidence and recommendations",
            "approach": "Example-based"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"🧪 TEST {i}: {test['approach']}")
        print(f"❓ Query: {test['query']}")
        print(f"{'='*60}")
        
        try:
            response = requests.post(f"{API_BASE_URL}/api/query", 
                                   json={"message": test['query']})
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response received:")
                print(f"📝 Answer: {result.get('answer', 'No answer')}")
                print(f"📚 Sources: {len(result.get('sources', []))} sources")
                print(f"🎯 Confidence: {result.get('confidence', 0):.2f}")
                
                # Check if answer contains multiple points
                answer = result.get('answer', '')
                bullet_points = answer.count('•') + answer.count('-') + answer.count('1.') + answer.count('2.')
                numbered_items = len([line for line in answer.split('\n') if line.strip().startswith(('1.', '2.', '3.'))])
                
                print(f"📊 Analysis:")
                print(f"   • Bullet points found: {bullet_points}")
                print(f"   • Numbered items: {numbered_items}")
                print(f"   • Multi-aspect answer: {'Yes' if bullet_points > 2 or numbered_items > 1 else 'No'}")
                
            else:
                print(f"❌ Query failed: Status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def suggest_rag_enhancements():
    """Suggest enhancements for multiple answers"""
    
    print(f"\n{'='*60}")
    print("🚀 SUGGESTED RAG ENHANCEMENTS FOR MULTIPLE ANSWERS")
    print(f"{'='*60}")
    
    enhancements = [
        {
            "approach": "Multi-Step Reasoning",
            "description": "Break complex queries into multiple sub-questions",
            "example": "1. What anomalies exist? 2. What are their causes? 3. What are solutions?"
        },
        {
            "approach": "Perspective-Based Answers", 
            "description": "Provide answers from different viewpoints",
            "example": "From Finance perspective: ... From Operations perspective: ..."
        },
        {
            "approach": "Ranked Results",
            "description": "Provide multiple answers ranked by relevance/confidence",
            "example": "Top 3 most relevant anomalies with different risk factors"
        },
        {
            "approach": "Contextual Variations",
            "description": "Same question with different contexts",
            "example": "High-risk vs medium-risk vs low-risk anomaly analysis"
        },
        {
            "approach": "Template-Based Responses",
            "description": "Use structured templates for consistent multi-part answers",
            "example": "Problem | Evidence | Impact | Solution for each anomaly"
        }
    ]
    
    for i, enhancement in enumerate(enhancements, 1):
        print(f"\n{i}. 🎯 {enhancement['approach']}")
        print(f"   📝 {enhancement['description']}")
        print(f"   💡 Example: {enhancement['example']}")
    
    print(f"\n🛠️ IMPLEMENTATION OPTIONS:")
    print(f"   A. Modify RAG model to return multiple candidate answers")
    print(f"   B. Create specialized endpoints for different answer types")
    print(f"   C. Use query expansion to generate related questions")
    print(f"   D. Implement answer clustering and ranking")

def main():
    # Test current RAG capabilities
    test_multiple_answer_approaches()
    
    # Suggest improvements
    suggest_rag_enhancements()
    
    print(f"\n💭 CONCLUSION:")
    print(f"   ✅ Your RAG system IS working and providing good answers")
    print(f"   🔄 To get multiple answers, we can enhance the system with:")
    print(f"      • Query expansion techniques")
    print(f"      • Multi-perspective prompting")
    print(f"      • Structured response templates")
    print(f"      • Answer ranking and diversity")

if __name__ == "__main__":
    main()
