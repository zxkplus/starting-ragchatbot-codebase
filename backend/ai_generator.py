from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


class AIGenerator:
    """Handles interactions with DeepSeek API via LangChain for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=0.01,  # DeepSeek rejects temperature=0
            max_tokens=800,
        )

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use (OpenAI function format)
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Build messages list
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=query),
        ]

        # Bind tools if available
        if tools:
            llm = self.llm.bind_tools(tools)
        else:
            llm = self.llm

        # Get response from DeepSeek
        try:
            response = llm.invoke(messages)
        except Exception as e:
            return f"Error generating response: {str(e)}"

        # Handle tool execution if needed
        if response.tool_calls and tool_manager:
            return self._handle_tool_execution(response, messages, tool_manager)

        # Return direct response
        return response.content

    def _handle_tool_execution(self, initial_response, messages: List, tool_manager) -> str:
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The AIMessage containing tool calls
            messages: Current message list (will be extended in place)
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Add AI's tool call response to messages
        messages.append(initial_response)

        # Execute all tool calls and collect results
        for tool_call in initial_response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            # Execute the tool
            tool_result = tool_manager.execute_tool(tool_name, **tool_args)

            # Add tool result as ToolMessage
            messages.append(ToolMessage(content=tool_result, tool_call_id=tool_id))

        # Get final response (without tools to avoid infinite tool-calling loops)
        try:
            final_response = self.llm.invoke(messages)
        except Exception as e:
            return f"Error generating response after tool execution: {str(e)}"

        return final_response.content
