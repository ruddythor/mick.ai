import os
import tempfile

from autogen import ConversableAgent, UserProxyAgent, AssistantAgent, GroupChat, config_list_from_json, GroupChatManager
from autogen.coding import LocalCommandLineCodeExecutor
from rag import find_doc


config_list = config_list_from_json(
    "OAI_CONFIG_LIST.json",
    # filter_dict={"tags": ["llama3"]},  # comment out to get all
)
print("\n\n*******\nConfig_list is: ", config_list)
# Create a temporary directory to store the code files.
temp_dir = tempfile.TemporaryDirectory()

# Create a local command line code executor.
code_executor = LocalCommandLineCodeExecutor(
    timeout=1000,  # Timeout for each code execution in seconds.
    work_dir="workspace",  # Use the temporary directory to store the code files.
    functions=[find_doc,]
)

user = UserProxyAgent(
    "user",
    max_consecutive_auto_reply=10,
    system_message="""Your name is User. 
        You are a helpful user proxy agent who can act on behalf of a user to provide feedback to other agents. 
        You are able to END conversation chains when you see the phrase "TERMINATE" or "#####TERMINATE" in a response to you, 
        although you can also continue the conversation if you feel your original request has not been addressed.
    """,
    llm_config={"config_list": config_list},
    human_input_mode="ALWAYS",
    code_execution_config=False, #{
        # 'work_dir': 'workspace',
        # 'use_docker': False,
        # 'last_n_messages': 15,
        # 'executor': code_executor,
    # },
    is_termination_msg=lambda msg: "\n#####TERMINATE" in msg["content"].upper(),
    default_auto_reply='I did not execute any code or generate any output. Please rephrase your response so that I can better understand.',
    description='A user-proxy agent to act on behalf of the user and relay results back to other agents.'
    # max_consecutive_auto_reply=
    #is_termination_msg=lambda msg: "\n#####TERMINATE" in msg["content"].upper(),
)

code_runner = AssistantAgent(
    "code_execution_agent",
    max_consecutive_auto_reply=10,
    system_message="""Your name is Code Executor. 
        Your ONLY task is to execute code from a coding writer and report the results back. You will always use skills from your functions library, accessible only to you.
        You will report ONLY the results of the code back, and you will never interpet or conjecture about the results.
    """,
    llm_config={"config_list": config_list},
    human_input_mode="NEVER",
    code_execution_config={
        # 'work_dir': 'workspace',
        # 'use_docker': False,
        'last_n_messages': 15,
        # 'work_dir': 'workspace',
        'executor': code_executor,
    },
    # is_termination_msg=lambda msg: "\n#####TERMINATE" in msg["content"].upper(),
    default_auto_reply='I did not execute any code or generate any output. Please rephrase your response so that I can better understand.',
    description='A code executing agent to execute code on behalf of the user and relay results back to other agents.'
    # max_consecutive_auto_reply=
    #is_termination_msg=lambda msg: "\n#####TERMINATE" in msg["content"].upper(),
)

# Register the tool function with the user proxy agent.
code_runner.register_for_execution(name="find_doc")(find_doc)


write_sys_msg = """
You have been given coding capability to solve tasks using Python code.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
    3. Always import your code from the 'functions' library.
    4. Never ever provide an isolated snippet of code, but instead you will provide full executable scripts, preferably in Python. Do not under any circumstances suggest another agent modify any code.
    5. You can provide access to your skill, or tool, by providing code like the following to the user: 
        ```python
        from functions import find_doc
        result = find_doc(query)
        print(result)
        ```
        Where `query` is the argument you, the code_execution_agent, will pass into the function.
    6. You have the following tools available to you: 
        functions.find_doc
"""
code_writer = ConversableAgent(
    "code_writer",
    max_consecutive_auto_reply=10,
    system_message=write_sys_msg,
    llm_config={"config_list": config_list},
    human_input_mode="NEVER",
    code_execution_config=False, #{
        # 'work_dir': 'workspace',
        # 'use_docker': False,
        # 'last_n_messages': 15,
        # 'executor': code_executor,
    # },
    # is_termination_msg=lambda msg: "\n#####TERMINATE" in msg["content"].upper(),    # function_map={'find_docs': find_doc},
    description='A primary code writer agent that can generate code and provide tools in its inventory to another agent for execution.'
)

# Register the tool function with the user proxy agent.
#code_writer.register_for_execution(name="find_doc")(find_doc)

def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user:
        # init -> retrieve
        return code_writer
    elif last_speaker is code_writer:
        # retrieve: action 1 -> action 2
        return code_runner
    elif last_speaker is code_runner:
        if "exitcode: 1" in messages[-1]["content"].lower():
            # retrieve --(execution failed)--> retrieve
            return code_writer
        else:
            # retrieve --(execution success)--> research
            return user
    elif last_speaker == "user":
        # research -> end
        return None

groupchat = GroupChat(
    agents=[user, code_writer, code_runner],
    messages=[],
    max_round=10,
    speaker_selection_method=state_transition,
    send_introductions=True,
)
manager = GroupChatManager(groupchat=groupchat, llm_config={'config_list': config_list})

# result = user.initiate_chat(
#     code_writer, message="Please supply me with code for the executor to use given the query 'YOUR QUERY HERE' for our find_doc tool/skill/function."
# )
while True:
    query = input("Please enter your local docs query: \n")
    result = user.initiate_chat(manager, message=f"Please supply me with code for the executor to use given the following query: \"{query}\"")
    print(result)