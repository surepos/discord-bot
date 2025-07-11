import discord
from discord import app_commands
from discord.ext import commands
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()


intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

def split_message(content: str, max_length: int = 2000) -> list:
    return [content[i:i + max_length] for i in range(0, len(content), max_length)]

@bot.event
async def on_ready():
    print(f'✅ Logged in as {bot.user}')
    try:
        synced = await bot.tree.sync()
        print(f"🔧 Synced {len(synced)} slash command(s).")
    except Exception as e:
        print(f"❌ Slash command sync failed: {e}")

@bot.tree.command(name="ask", description="Ask a question from the FAQ bot.")
@app_commands.describe(question="What would you like to know?")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)

    try:
        embeddings = OpenAIEmbeddings()
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o-mini",
            api_version="2025-01-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        vectorstore = PineconeVectorStore(
            embedding=embeddings, index_name=os.environ["INDEX_NAME"]
        )

        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm, prompt=retrieval_qa_chat_prompt
        )
        retrieval_chain = create_retrieval_chain(
            retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
        )
        response = retrieval_chain.invoke(input={"input": question})

        chunks = split_message(response["answer"])
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)

    except Exception as e:
        await interaction.followup.send(f"⚠️ Error: {str(e)}")

bot.run(os.getenv("DISCORD_TOKEN"))
