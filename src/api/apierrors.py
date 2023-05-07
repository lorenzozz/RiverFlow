
class RequestError(Exception):
    """ Errore di richiesta incorretta a una API. È possibile accedere
    alla richiesta che ha portato all'errore tramite il membro
    RequestError.failed_request """

    def __init__(self, reason, failed_request):
        self._reason = reason
        self.failed_request = failed_request

    def __str__(self):
        return self._reason