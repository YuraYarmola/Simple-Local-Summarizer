from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


class Summarizer:
    def __init__(self, chunk_size=6000):
        # init of local llm to make summarization
        self.pipe = pipeline("summarization", model="facebook/bart-large-cnn")
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
        self.chunk_size = chunk_size

    # function to divide text to chunks
    def split_document(self, documents_text: list[str]) -> list[Document]:
        documents = []
        for text in documents_text:
            documents.append(Document(text))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        return split_docs

    # function to make summary by langchain, of any length
    def summarize(self, text):
        split_docs = self.split_document([text])
        prompt_template = """Write a concise summary of the following transcribed text:
                {text}
                CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)

        refine_template = (
            "Your job is to summarize the part of transcribed text as best as you can\n"
            "We have provided an existing summary up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing summary"
            "(only if needed) with some more context below.\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "Given the new context, refine the original summary in English"
            "If the context isn't useful, return the original summary."
        )
        refine_prompt = PromptTemplate.from_template(refine_template)
        return self.run_refine_chain(prompt=prompt, refine_prompt=refine_prompt, split_docs=split_docs)

    # function to run refine chain
    def run_refine_chain(self, prompt: PromptTemplate, refine_prompt: PromptTemplate,
                         split_docs: list[Document], **kwargs):
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )
        argument = {"input_documents": split_docs}
        argument.update(kwargs)
        chain_result = chain(argument, return_only_outputs=True)

        return chain_result["output_text"]


if __name__ == '__main__':
    # TEST SUMMARIZATION
    ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
    2010 marriage license application, according to court documents.
    Prosecutors said the marriages were part of an immigration scam.
    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
    After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
    Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
    All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
    Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
    Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
    The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
    Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
    Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
    If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """
    print(Summarizer().summarize(ARTICLE))
