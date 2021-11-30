pub(crate) fn parse_put_actions(body: &Body) -> Result<ParsedRequest, Error> {
    METRICS.put_api_requests.actions_count.inc();
    let action_body = serde_json::from_slice::<ActionBody>(body.raw()).map_err(|e| {
        METRICS.put_api_requests.actions_fails.inc();
        Error::SerdeJson(e)
    })?;
    match action_body.action_type {
        ActionType::FlushMetrics => Ok(ParsedRequest::new_sync(VmmAction::FlushMetrics)),
        ActionType::InstanceStart => Ok(ParsedRequest::new_sync(VmmAction::StartMicroVm)),
        ActionType::SendCtrlAltDel => {
            #[cfg(target_arch = "aarch64")]
            return Err(Error::Generic(
                StatusCode::BadRequest,
                "SendCtrlAltDel does not supported on aarch64.".to_string(),
            ));
            #[cfg(target_arch = "x86_64")]
            Ok(ParsedRequest::new_sync(VmmAction::SendCtrlAltDel))
        }
    }
}
