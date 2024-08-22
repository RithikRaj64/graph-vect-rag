class StorageContextNotFoundException(Exception):
    pass


class LLMNotFoundException(Exception):
    pass


class EmbeddingModelNotFoundException(Exception):
    pass


class InvlaidModelIdException(Exception):
    pass


class KnowledgeBaseAlreadyExists(Exception):
    pass


class KnowledgeBaseNotFound(Exception):
    pass


class KnowledgeBaseNotConfigured(Exception):
    pass
