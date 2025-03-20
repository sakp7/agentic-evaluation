__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew,LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from litellm import completion
import json

# Load environment variables
load_dotenv()

# PDF file path
PDF_PATH = "chapter5.pdf"

# Initialize Streamlit app
def main():
    st.title("Student Answer Evaluation System")
   
    # Sidebar for mode selection
    st.sidebar.title("Options")
    mode = st.sidebar.radio("Select Mode", ["Basic RAG Evaluation", "Agent Orchestration Evaluation"])
   
    # Check if PDF exists
    if not os.path.exists(PDF_PATH):
        st.error(f"PDF file not found: {PDF_PATH}")
        st.info(f"Please place a file named '{PDF_PATH}' in the same directory as this script.")
        return
   
    st.success(f"Using PDF: {PDF_PATH}")
   
    # Initialize RAG components
    with st.spinner("Initializing RAG system..."):
        retriever = initialize_retriever(PDF_PATH)

    # Input fields
    question = st.text_area("Question", """ In  a  flower  bed,  there  are  23  rose  plants  in  the  first  row,  21  in  the
second, 19 in the third, and so on. There are 5 rose plants in the last row. How many
rows are there in the flower bed?""", height=100)
    
    student_answer = st.text_area("Student Answer", value="""The number of rose plants in the 1st, 2nd, 3rd, . . ., rows are :
23, 21, 19, . . ., 5
It forms an AP (Why?). Let the number of rows in the flower bed be n.
Then a = 23, d = 21 – 23 = – 2, an = 5
As, an  = a + (n – 1) d
We have, 5 = 23 + (n – 1)(– 2)
i.e., – 18 = (n – 1)(– 2)
i.e., n = 10
So, there are 10 rows in the flower bed.""", height=150)
   
    if st.button("Evaluate") and question and student_answer:
        with st.spinner("Retrieving relevant context..."):
            # Get relevant chunks from PDF using the retriever
            relevant_chunks = retrieve_context(retriever, question, k=5)
           
            if relevant_chunks:
                st.success("Retrieved relevant information from PDF")
               
                # Show relevant chunks in an expandable section
                with st.expander("View Retrieved Context"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.markdown(f"**Chunk {i+1}:**\n{chunk}")
               
                # Evaluate the student's answer based on the selected mode
                if mode == "Basic RAG Evaluation":
                    with st.spinner("Evaluating student answer using basic RAG..."):
                        evaluation = evaluate_answer_without_agents(question, student_answer, relevant_chunks)
                        
                        # Display evaluation results
                        st.subheader("Basic RAG Evaluation Results")
                        st.markdown(evaluation)
               
                elif mode == "Agent Orchestration Evaluation":
                    with st.spinner("Evaluating student answer using agent orchestration..."):
                        evaluation = evaluate_answer_with_agents(question, student_answer, relevant_chunks)
                        
                        # Display evaluation results
                        st.subheader("Agent Orchestration Evaluation Results")
                        st.markdown(evaluation)
            else:
                st.error("Could not retrieve relevant information from the PDF.")

# Function to initialize the retriever
def initialize_retriever(pdf_path):
    try:
        # Load and split document
        docs = PyMuPDFLoader(pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1048, chunk_overlap=150)
        split_docs = splitter.split_documents(docs)
       
        # Initialize embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(
            model='models/text-embedding-004',
            google_api_key=os.getenv('GOOGLE_API_KEY')
        )
       
        # Create vector store and retriever
        vectorstore = FAISS.from_documents(documents=split_docs, embedding=embedding_model)
        return vectorstore.as_retriever(search_kwargs={"k": 5})
   
    except Exception as e:
        st.error(f"Error initializing retriever: {str(e)}")
        return None

# Function to retrieve context from PDF
def retrieve_context(retriever, question, k=5):
    try:
        # Get relevant documents
        docs = retriever.get_relevant_documents(question)
       
        # Extract and return the content
        return [doc.page_content for doc in docs[:k]]
   
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return []

# Function to evaluate the student's answer without agents
def evaluate_answer_without_agents(question, student_answer, context_chunks):
    try:
        # Combine context chunks into a single context
        combined_context = "\n\n".join(context_chunks)
       
        # Create improved evaluation prompt with structured output
        evaluation_prompt = f"""
        You are an experienced teacher evaluating a student's answer based on reference material.
       
        Question: {question}
       
        Student Answer: {student_answer}
       
        Reference Material:
        {combined_context}
       
        Please evaluate the student's answer and provide your analysis in JSON format with the following structure:
        {{
            "score": <score out of 10>,
            "correct_points": ["<point 1>", "<point 2>", ...],
            "incorrect_points": ["<point 1>", "<point 2>", ...],
            "gap_analysis": ["<concept 1>", "<concept 2>", ...],
            "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>", ...]
        }}
       
        Base your evaluation ONLY on the reference material provided. If the reference material doesn't contain information to evaluate the answer, say so.
        """
       
        # Use LiteLLM to get evaluation
        response = completion(
            model="groq/llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an expert teacher providing detailed evaluation of student answers. Return your analysis in JSON format."},
                {"role": "user", "content": evaluation_prompt}
            ]
        )
       
        # Extract content from response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            evaluation_content = response.choices[0].message.content
            
            # Try to parse as JSON and format it nicely
            try:
                # Find JSON content if wrapped in markdown code blocks or other text
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```|(\{.*\})', evaluation_content, re.DOTALL)
                
                if json_match:
                    json_str = next(group for group in json_match.groups() if group)
                    evaluation_data = json.loads(json_str)
                    
                    # Format the output in a structured way
                    correct_points = "\n".join([f"- {point}" for point in evaluation_data['correct_points']])
                    incorrect_points = "\n".join([f"- {point}" for point in evaluation_data['incorrect_points']])
                    gap_analysis = "\n".join([f"- {gap}" for gap in evaluation_data['gap_analysis']])
                    improvement_suggestions = "\n".join([f"- {suggestion}" for suggestion in evaluation_data['improvement_suggestions']])

                    formatted_output = f"""
                    ## Score: {evaluation_data['score']}/10

                    ### Correct Points:
                    {correct_points}

                    ### Incorrect Points:
                    {incorrect_points}

                    ### Gap Analysis:
                    {gap_analysis}

                    ### Improvement Suggestions:
                    {improvement_suggestions}
                    """
                    return formatted_output
            except:
                pass
            
            # If JSON parsing fails, return the raw content
            return evaluation_content
        else:
            return str(response)
   
    except Exception as e:
        return f"Error evaluating answer: {str(e)}"

# Function to evaluate the student's answer with CrewAI agents
def evaluate_answer_with_agents(question, student_answer, context_chunks):
    try:
        # Combine context chunks into a single context
        combined_context = "\n\n".join(context_chunks)
        
        # Define the model for all agents
        llm =LLM(model="groq/llama-3.3-70b-versatile",temperature=0.2)
    
        
        # Define the solution extraction agent
        solution_agent = Agent(
            role='Solution Expert',
            goal='Extract the correct solution from the reference material',
            backstory='You are an expert in extracting and understanding solutions to mathematical problems from educational materials.',
            llm=llm,
            verbose=True
        )
        
        # Define the evaluation agent
        evaluation_agent = Agent(
            role='Evaluation Expert',
            goal='Evaluate student answers against correct solutions',
            backstory='You are an experienced educator who specializes in evaluating student work and providing constructive feedback.',
            llm=llm,
            verbose=True
        )
        
        # Define the gap analysis agent
        gap_analysis_agent = Agent(
            role='Gap Analysis Expert',
            goal='Identify knowledge gaps in student understanding',
            backstory='You are specialized in identifying conceptual gaps and misunderstandings in student work.',
            llm=llm,
            verbose=True
        )

        # Define tasks for each agent
        extract_solution_task = Task(
            description=f"""
            Review the following reference material and extract the correct solution for this question:
            
            Question: {question}
            
            Reference Material:
            {combined_context}
            
            Output the correct approach and solution as detailed as possible in JSON format with the following structure:
            {{
                "correct_approach": "step by step explanation",
                "correct_solution": "final answer with explanation",
                "key_concepts": ["concept1", "concept2", ...]
            }}
            """,
            agent=solution_agent,
            expected_output="JSON with correct approach, solution, and key concepts"
        )
        
        evaluate_answer_task = Task(
        description=f"""
        Compare the student's answer with the correct solution and evaluate it.Give the reponse to student like say You have done ... etc 
        
        Question: {question}
        
        Student Answer: {student_answer}
        
        Use the correct solution from the previous task and provide your evaluation in JSON format with the following structure:
        {{
            "score": <score out of 10>,
            "approach_correct": <true or false>,
            "result_correct": <true or false>,
            "concepts_all_included": <true or false>,
            "missing_concepts": ["concept1", "concept2", ...],
            "correct_points": ["point1", "point2", ...],
            "incorrect_points": ["point1", "point2", ...],
            "overall_assessment": "summary of evaluation"
        }}
        """,
        agent=evaluation_agent,
        expected_output="JSON with evaluation details including approach correctness, result correctness, and concept coverage",
        context=[extract_solution_task]
    )
        
        gap_analysis_task = Task(
            description=f"""
            Based on the evaluation and correct solution, identify knowledge gaps and provide improvement suggestions.Giev response like You have done good work not student have done good work
            
            Question: {question}
            
            Student Answer: {student_answer}
            
            Provide your analysis in JSON format with the following structure:
            {{
                "knowledge_gaps": ["gap1", "gap2", ...],
                "misconceptions": ["misconception1", "misconception2", ...],
                "improvement_suggestions": ["suggestion1", "suggestion2", ...],
                "learning_resources": ["resource1", "resource2", ...]
            }}
            """,
            agent=gap_analysis_agent,
            expected_output="JSON with knowledge gaps, misconceptions, improvement suggestions, and learning resources",
            context=[extract_solution_task, evaluate_answer_task]
        )

        # Define the crew
        crew = Crew(
            agents=[solution_agent, evaluation_agent, gap_analysis_agent],
            tasks=[extract_solution_task, evaluate_answer_task, gap_analysis_task],
            verbose=True
        )
        
        # Kickoff the crew
        # Modify the kickoff line and add task output collection
        result = crew.kickoff()

        # Get outputs from each task
        # Get outputs from each task
        solution_output_raw = extract_solution_task.output
        evaluation_output_raw = evaluate_answer_task.output
        gap_analysis_output_raw = gap_analysis_task.output

        # Add debug output to help troubleshoot
        print(f"Solution output type: {type(solution_output_raw)}")
        print(f"Evaluation output type: {type(evaluation_output_raw)}")
        print(f"Gap analysis output type: {type(gap_analysis_output_raw)}")

        # Parse outputs with improved extraction function
        solution_output = extract_json_from_output(solution_output_raw)
        evaluation_output = extract_json_from_output(evaluation_output_raw)
        gap_analysis_output = extract_json_from_output(gap_analysis_output_raw)
                
        # Process the result
        try:
            # Extract solutions from task outputs
            # Prepare the approach, result, and concepts assessment
            approach_status = "✅ Correct" if evaluation_output.get('approach_correct', False) else "❌ Incorrect"
            result_status = "✅ Correct" if evaluation_output.get('result_correct', False) else "❌ Incorrect"
            concepts_status = "✅ All included" if evaluation_output.get('concepts_all_included', False) else "❌ Some missing"

            # Format missing concepts if any
            missing_concepts = "\n".join([f"- {concept}" for concept in evaluation_output.get('missing_concepts', [])])
            if not missing_concepts and not evaluation_output.get('concepts_all_included', False):
                missing_concepts = "- No specific missing concepts identified"

            # Pre-compute formatted strings
            correct_points = "\n".join([f"- {point}" for point in evaluation_output.get('correct_points', ['No correct points identified'])])
            incorrect_points = "\n".join([f"- {point}" for point in evaluation_output.get('incorrect_points', ['No incorrect points identified'])])
            improvement_suggestions = "\n".join([f"- {suggestion}" for suggestion in gap_analysis_output.get('improvement_suggestions', ['No suggestions provided'])])
            # recommended_resources = "\n".join([f"- {resource}" for resource in gap_analysis_output.get('learning_resources', ['No resources recommended'])])

            formatted_output = f"""
            # Evaluation Results

            ## Score: {evaluation_output.get('score', 'N/A')}/10

            ## Assessment Summary
            - Approach: {approach_status}
            - End Result: {result_status}
            - Concepts Used: {concepts_status}

            ## Correct Solution
            {solution_output.get('correct_approach', 'No correct approach found')}

            Final Answer: {solution_output.get('correct_solution', 'No solution found')}

            ## Student Performance

            ### What the student got right:
            {correct_points}

            ### What the student got wrong:
            {incorrect_points}

            ## Concept Analysis
            ### Missing Concepts:
            {missing_concepts}

            ## Improvement Plan
            ### Suggestions:
            {improvement_suggestions}

          
            """
            return formatted_output
            
        except Exception as e:
            # If parsing fails, return the raw output
            return f"Raw agent output (parsing failed with error: {str(e)}):\n\n{result}"
   
    except Exception as e:
        return f"Error evaluating answer with agents: {str(e)}"

# Helper function to extract JSON from agent outputs
# Update the extract_json_from_output function to handle TaskOutput objects
def extract_json_from_output(output):
    if output is None:
        return {}
        
    # Handle TaskOutput objects
    if hasattr(output, 'raw_output'):
        output = output.raw_output
    elif not isinstance(output, str):
        try:
            # Try to convert to string
            output = str(output)
        except:
            return {}
    
    try:
        # Try direct JSON parsing
        return json.loads(output)
    except:
        # Look for JSON in markdown code blocks or surrounded text
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```|(\{.*\})', output, re.DOTALL)
        
        if json_match:
            try:
                json_str = next(group for group in json_match.groups() if group)
                return json.loads(json_str)
            except:
                pass
    
    # Return empty dict if all parsing attempts fail
    return {}
    
    # Return empty dict if all parsing attempts fail
    return {}

if __name__ == "__main__":
    main()
