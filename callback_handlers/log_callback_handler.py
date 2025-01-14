from typing import Any, Dict, List, Union
import logging

from langchain.schema.output import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages.base import BaseMessage


class LogCallbackHandler(BaseCallbackHandler):
    '''Langchain callback handler that logs agent and chain executions.'''

    def __init__(self, name: str, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._num_running_chains = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        '''Run when LLM starts running.'''
        pass

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        '''Run when Chat Model starts running.'''
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        '''Run on new LLM token. Only available when streaming is enabled.'''
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        '''Run when LLM ends running.'''
        pass

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        '''Run when LLM errors.'''
        self.logger.exception(error)

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        '''Run when chain starts running.'''
        self._num_running_chains += 1
        self.logger.info('Entering new LLM chain ({} total)'.format(
            self._num_running_chains))
        self.logger.debug('Inputs: {}'.format(inputs))

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        '''Run when chain ends running.'''
        self._num_running_chains -= 1
        self.logger.info('Ended LLM chain ({} total)'.format(
            self._num_running_chains))
        self.logger.debug('Outputs: {}'.format(outputs))

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        '''Run when chain errors.'''
        self._num_running_chains -= 1
        self.logger.error(
            'LLM chain encountered an error ({} running): {}'.format(
                self._num_running_chains, error))

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        '''Run when tool starts running.'''
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        '''Run when tool ends running.'''
        pass

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        '''Run when tool errors.'''
        pass

    def on_text(self, text: str, **kwargs: Any) -> Any:
        '''Run on arbitrary text.'''
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        '''Run on agent action.'''
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        '''Run on agent end.'''
        pass
