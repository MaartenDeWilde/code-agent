// Several tools read from and write to the console (confirmations, ask_user),
// which tests redirect via Console.SetIn — global state that must not be
// touched from two tests at once.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
