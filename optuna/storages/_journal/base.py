import abc
from typing import Any
from typing import Dict
from typing import List


class BaseJournalLogStorage(abc.ABC):
    """Base class for Journal storages.

    Storage classes implementing this base class must guarantee process safety. This means,
    multiple processes might concurrently call ``read_logs`` and ``append_logs``. If the
    backend storage does not internally support mutual exclusion mechanisms, such as locks,
    you might want to use :class:`~optuna.storages.JournalFileSymlinkLock` or
    :class:`~optuna.storages.JournalFileOpenLock` for creating a critical section.

    """

    @abc.abstractmethod
    def read_logs(self, log_number_from: int) -> List[Dict[str, Any]]:
        """Read logs with a log number greater than or equal to ``log_number_from``.

        If ``log_number_from`` is 0, read all the logs.

        Args:
            log_number_from:
                A non-negative integer value indicating which logs to read.

        Returns:
            Logs with log number greater than or equal to ``log_number_from``.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        """Append logs to the backend.

        Args:
            logs:
                A list that contains json-serializable logs.
        """

        raise NotImplementedError
