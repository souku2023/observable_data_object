"""
An Observer Pattern that mimics the LiveData of Android.

@author Soukumarya Saha (sahasoukumarya@gmail.com)
"""
import atexit
import threading
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, final


# Exceptions
# noinspection PyMissingOrEmptyDocstring
class FailedToAddObserverError(Exception):
    pass


# noinspection PyMissingOrEmptyDocstring
class FailedToNotifyObserverError(Exception):
    pass


# noinspection PyMissingOrEmptyDocstring
class FailedToCreateObservableError(Exception):
    pass


# noinspection PyMissingOrEmptyDocstring
class HandlerAlreadySetError(Exception):
    pass


# noinspection PyMissingOrEmptyDocstring
class ThreadPoolNotInitializedError(Exception):
    pass


NotificationErrorHandler = Callable[[Exception, str, Callable, Any], Any]


@final
class ObservableDataObject(object):
    """
    Implements the observer pattern for notifying multiple observers of data
    changes. Uses a static thread pool shared across all instances for observer
    notification.
    """

    # Static thread pool shared by all instances
    __threadpool: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=10)
    __shutdown_registered: bool = False

    def __init__(self, data: Any):
        """
        Implements the observer pattern for notifying multiple observers of data
        changes. Uses a static thread pool shared across all instances for
        observer notification.

        Initialize the observable data object with initial data.

        :param data: The initial data value
        :type data: Any

        :raises FailedToCreateObservableError: If initial data is None
        """
        if data is None:
            raise FailedToCreateObservableError(
                "Initialization data type cannot be None."
            )

        super().__init__()

        self.__data = data
        self.__data_type = type(data)
        self.__observers = []
        self.__observer_names = []
        self.__observer_lock = threading.RLock()
        self.__value_lock = threading.RLock()
        self.__error_handler = None
        self.__notification_task_list: list[Future] = list()

        # Register shutdown for thread pool
        if not ObservableDataObject.__shutdown_registered:
            atexit.register(ObservableDataObject.shutdown)

    def observe(self, observer: Callable, observer_name: str | None = None):
        """
        Register an observer function to be notified of data changes.

        :param observer: Callable function to be notified of changes
        :type observer: Callable
        :param observer_name: Optional name for the observer (for logging)
        :type observer_name: str | None

        :raises FailedToAddObserverError: If observer is not callable
        """
        if not callable(observer):
            raise FailedToAddObserverError("Observer is not a Callable.")

        with self.__observer_lock:
            self.__observers.append(observer)
            self.__observer_names.append(observer_name)

    def post_value(self, new_data: Any):
        """
        Post a new value to notify all observers. A ``post_value`` call will
        trigger all the observers for one value before triggering the observers
        for the next call.

        :param new_data: New data value to notify observers about. If new_data
            is None, handle the observers accordingly
        :type new_data: Any

        :raises FailedToNotifyObserverError: If data type doesn't match
        initial type
        """
        if type(new_data) != self.__data_type and new_data is not None:
            raise FailedToNotifyObserverError(
                f"Posted data should be of type '{self.__data_type}' or "
                f"'None', not {type(new_data)}'"
            )

        if ObservableDataObject.__threadpool is None:
            raise ThreadPoolNotInitializedError("Thread pool not initialized")

        # Update the current data
        self.__data = new_data

        # Create a snapshot of current observers
        with self.__observer_lock:
            observers = self.__observers[:]
            names = self.__observer_names[:]

        # Notify all observers concurrently using the shared thread pool
        for idx, observer in enumerate(observers):
            observer_name = names[idx] if names[idx] is not None else repr(observer)
            # Trigger all observers for a value before triggering observers for
            # the next post_value call
            with self.__value_lock:
                try:
                    task: Future = ObservableDataObject.__threadpool.submit(
                            self.__notify_observer,
                            observer, new_data, observer_name
                    )
                    self.__notification_task_list.append(task)
                    task.add_done_callback(
                            self.__remove_from_notification_task_list
                    )
                except Exception as e:
                    raise FailedToNotifyObserverError(
                        "Failed to add observer to thread pool"
                    ) from e

    def __notify_observer(
        self, observer: Callable, new_data: Any, observer_name: str
    ) -> None:
        """
        Notify a single observer of new data.

        :param observer: Observer function to call
        :type observer: Callable
        :param new_data: New data value to pass to observer
        :type new_data: Any
        :param observer_name: Name of observer for logging
        :type observer_name: str
        """
        try:
            observer(new_data)
        except Exception as e:
            # Notify error in the same thread
            if self.__error_handler:
                self.__error_handler(e, observer_name, observer, new_data)
            raise e

    def detach_observer(self, observer: Callable):
        """
        Remove an observer from the notification list.

        :param observer: Observer function to remove
        :type observer: Callable
        """
        with self.__observer_lock:
            if observer in self.__observers:
                idx = self.__observers.index(observer)
                del self.__observers[idx]
                del self.__observer_names[idx]

    @classmethod
    def shutdown(cls):
        """
        Shutdown the shared thread pool. Should be called when the
        application exits.
        """
        cls.__threadpool.shutdown(wait=True)
        cls.__threadpool = None

    def on_error(self, handle_error: NotificationErrorHandler) -> None:
        """
        A method that is triggered when a handler fails.

        NOTE: **DO NOT** call post value of same observable in the
        error handler!

        :param handle_error: Method to be called when the handler fails
        :type handle_error: NotificationErrorHandler
        :return: None
        """
        if self.__error_handler is None:
            self.__error_handler = handle_error
            return

        raise HandlerAlreadySetError("Error handler already set")

    def __remove_from_notification_task_list(self, task: Future):
        """Removes the pending notification task"""
        self.__notification_task_list.remove(task)

    def destroy(self):
        """Cancels all the notification tasks and removes all observers"""
        for notification_task in self.__notification_task_list:
            notification_task.cancel()
        self.__notification_task_list.clear()
        self.__observers.clear()
        self.__observer_names.clear()
        self.__error_handler = None
